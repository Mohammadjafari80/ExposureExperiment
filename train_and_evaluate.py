from ast import arg
from asyncore import write
from distutils.command.config import config
import argsparser
import torch
import torch.nn as nn
from torch import optim
from torchvision.utils import save_image
from torchvision.utils import make_grid
from utills import auc_softmax, auc_softmax_adversarial, save_model_checkpoint, load_model_checkpoint, lr_schedule, get_visualization_batch, visualize, get_attack_name
from tqdm import tqdm
from torchattacks import FGSM, PGD
from models import Net
from constants import PGD_CONSTANT, dataset_labels
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.tensorboard.writer import SummaryWriter
from datasets import get_dataloader
import os


def run(model, checkpoint_path, train_attack, test_attacks, trainloader, testloader, writer, test_step, save_step, max_epochs, device, loss_threshold=1e-3):

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    init_epoch = 0

    checkpoint = load_model_checkpoint(model=model, optimizer=optimizer, path=checkpoint_path)

    if checkpoint is not None:
        model, optimizer, init_epoch, loss = checkpoint

    vis_batch_train = get_visualization_batch(dataloader=trainloader, n=50)
    vis_batch_test = get_visualization_batch(dataloader=testloader, n=50)

    for epoch in range(init_epoch, max_epochs+1):

        torch.cuda.empty_cache()

        if epoch % test_step == 0 :
                
                test_auc = {}
                test_accuracy = {}

                clean_auc, clean_accuracy  = auc_softmax(model=model, epoch=epoch, test_loader=testloader, device=device)
                test_auc['Clean'], test_accuracy['Clean'] = clean_auc, clean_accuracy

                for attack_name, attack in test_attacks.items():
                    adv_auc, adv_accuracy = auc_softmax_adversarial(model=model, epoch=epoch, test_loader=testloader, test_attack=attack, device=device)
                    test_auc[attack_name], test_accuracy[attack_name] = adv_auc, adv_accuracy

                writer.add_scalars('AUC-Test', test_auc, epoch)
                writer.add_scalars('Accuracy-Test', test_accuracy, epoch)
                writer.flush()

                for attack_name, attack in test_attacks.items():
                    writer.add_figures(f'Sample Peturbations Train {get_attack_name(attack)}', visualize(vis_batch_train[0], vis_batch_train[1], attack), epoch)
                    writer.add_figures(f'Sample Peturbations Test {get_attack_name(attack)}', visualize(vis_batch_test[0], vis_batch_test[1], attack), epoch)
                    writer.flush()

        torch.cuda.empty_cache()

        train_auc, train_accuracy, train_loss = train_one_epoch(epoch=epoch,\
                                                                max_epochs=max_epochs, \
                                                                model=model,\
                                                                optimizer=optimizer,
                                                                criterion=criterion,\
                                                                trainloader=trainloader,\
                                                                train_attack=train_attack,\
                                                                lr=0.1,\
                                                                device=device)
        
        writer.add_scalar('AUC-Train', train_auc, epoch)
        writer.add_scalar('Accuracy-Train', train_accuracy, epoch)
        writer.add_scalar('Accuracy-Loss', train_loss, epoch)
        writer.flush()
        

        if train_loss < loss_threshold:
            save_model_checkpoint(model=model, epoch=epoch, loss=train_loss, path=checkpoint, optimizer=optimizer)
            break
        
        if epoch > 0 and epoch % save_step == 0:
            save_model_checkpoint(model=model, epoch=epoch, loss=train_loss, path=checkpoint, optimizer=optimizer)

    writer.close()

def train_one_epoch(epoch, max_epochs, model, optimizer, criterion, trainloader, train_attack, lr, device): 

    soft = torch.nn.Softmax(dim=1)

    preds = []
    anomaly_scores = []
    true_labels = []
    running_loss = 0
    accuracy = 0

    model.train()
    with tqdm(trainloader, unit="batch") as tepoch:
        torch.cuda.empty_cache()
        for i, (data, target) in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch + 1}/{max_epochs}")

            updated_lr = lr_schedule(learning_rate=lr, t=epoch + (i + 1) / len(list(tepoch)), max_epochs=max_epochs) 
            optimizer.param_groups[0].update(lr=updated_lr)
            
            data, target = data.to(device), target.to(device)
            target = target.type(torch.LongTensor).cuda()
            
            # Adversarial attack on every batch
            data = train_attack(data, target)

            # Zero gradients for every batch
            optimizer.zero_grad()

            # Make predictions for this batch
            output = model(data)

            # Compute the loss and its gradients
            loss = criterion(output, target)
            loss.backward()

            # Adjust learning weights
            optimizer.step()
            
            true_labels += target.detach().cpu().numpy().tolist()

            predictions = output.argmax(dim=1, keepdim=True).squeeze()
            preds += predictions.detach().cpu().numpy().tolist()
            correct = (torch.tensor(preds) == torch.tensor(true_labels)).sum().item()
            accuracy = correct / len(preds)

            probs = soft(output).squeeze()
            anomaly_scores += probs[:, 1].detach().cpu().numpy().tolist()

            running_loss += loss.item() * data.size(0)

            tepoch.set_postfix(loss=running_loss / len(preds), accuracy=100. * accuracy)

    return  roc_auc_score(true_labels, anomaly_scores) , \
            accuracy_score(true_labels, preds, normalize=True), \
            running_loss / len(preds)


##################
#  Parsing Args  #
##################

args = argsparser.parse_args()
print(args)


################
#  Set Device  #
################

device = None

try:
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
except:
    raise ValueError('Wrong CUDA Device!')


print(device)

####################
#  Model Selection #
####################

model = None

try:
    model = Net(args.model).to(device)
except Exception as err:
    raise err

print(args.args.model)


#####################
#  Attacks Eps Init #
#####################

attack_eps = None

try:
    attack_eps = eval(args.attack_eps)
except:
    raise ValueError('Wrong Epsilon Value!')

######################
#  Test Attacks Init #
######################

# !python .\train_and_evaluate.py --test_attacks FGSM PGD-10 PGD-100

test_attacks = {}

for test_attack in args.test_attacks:
    try:
        attack_type = test_attack.split('-')[0] if test_attack != 'FGSM' else 'FGSM'
        if attack_type == 'FGSM':
            current_attack = FGSM(model, eps=attack_eps)
            current_attack.set_mode_targeted_least_likely()
            test_attacks[test_attack] = current_attack
        elif attack_type == 'PGD':
            steps = eval(test_attack.split('-')[1])
            alpha = (PGD_CONSTANT * attack_eps) / steps
            current_attack = PGD(model, eps=attack_eps, alpha=alpha, steps=steps)
            current_attack.set_mode_targeted_least_likely()
            test_attacks[test_attack] = current_attack
    except:
        raise ValueError('Invalid Attack Params!')


######################
#  Train Attack Init #
######################

train_steps = args.train_step
train_alpha = (PGD_CONSTANT * attack_eps) / train_steps
train_attack = PGD(model, eps=attack_eps, alpha=train_alpha, steps=train_steps)


################
#  Dataloaders #
################

trainloader, testloader = get_dataloader(normal_dataset=args.source_dataset, normal_class_indx=args.source_class, exposure_dataset=args.exposure_dataset, batch_size=args.batch_size)

#########################
#  init checkpoint path #
#########################

checkpoint_dir = os.path.join(args.checkpoints_path, f'normal-{args.source_dataset}', f'normal-class-{args.source_class:02d}-{dataset_labels[args.source_class]}', f'exposure-{args.exposure_dataset}')
checkpoint_name = f'{args.source_dataset}-{args.source_class:02d}--{args.exposure_dataset}.pt'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)


############################
#  init tensorboard writer #
############################

writer_dir = os.path.join(args.output_path, f'normal-{args.source_dataset}', f'normal-class-{args.source_class:02d}-{dataset_labels[args.source_class]}', f'exposure-{args.exposure_dataset}')
writer = SummaryWriter('runs/fashion_mnist_experiment_1')


##################################
#               RUN              #
##################################


run(model=model,\
    checkpoint_path=checkpoint_path,\
    train_attack=train_attack,\
    test_attacks=test_attacks,\
    trainloader=trainloader,\
    testloader=testloader,\
    writer=writer,\
    test_step=args.test_step,\
    save_step=args.save_step,\
    max_epochs=args.max_epochs,\
    device=device\
    loss_threshold=args.loss_threshold\
    )