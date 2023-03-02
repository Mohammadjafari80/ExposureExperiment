from argparse import ArgumentParser

def parse_args():
    """Command-line argument parser for training."""

    parser = ArgumentParser(description='Outlier Exposure Experiments Automation')

    parser.add_argument('--source_dataset', help='Target Dataset as one-class for normal',
                        choices=['cifar10', 'cifar100', 'mnist', 'fmnist', 'mvtec-ad', 'med'], type=str)

    parser.add_argument('--source_class', help='Index of Normal Class',
                        default=None, type=int)

    parser.add_argument('--output_path', help='Path to which plots, results, etc will be recorded',
                        default='./results', type=str)
                        
    parser.add_argument('--source_dataset_path', help='Path to source datasets, if not given, will be downloaded',
                        default=None, type=str)
                        
    parser.add_argument('--exposure_dataset_path', help='Path to which plots, results, etc will be recorded',
                        default=None, type=str)

    parser.add_argument('--exposure_dataset', help='Target Dataset as one-class for normal',
                        choices=['cifar10', 'cifar100', 'mnist', 'fmnist', 'mvtec-ad', 'med'], type=str)

    parser.add_argument("--checkpoints_path", help='Path to save the checkpoint of trained model',
                        default=None, type=str)

    parser.add_argument("--max_epochs", help='Maximum number of epochs to Continue training',
                        default=30, type=str)

    parser.add_argument('--attack_type', help='Desired Attack for adversarial setting',
                        choices=['PGD', 'FGSM'], type=str)

    
    parser.add_argument('--test_step', help='If given x, every x step a test would be performed',
                        default=1, type=int)

    parser.add_argument('--model', help='Model architecture',
                        choices=['resnet18', 'preactresnet18', 'pretrained_resnet18', 'vit'], default='preactresnet18', type=str)
    

    return parser.parse_args()