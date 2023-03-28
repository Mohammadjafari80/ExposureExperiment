
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, SVHN, FashionMNIST
import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
from utills import sparse2coarse
from constants import CIFAR10_PATH, CIFAR100_PATH, MNIST_PATH, FMNIST_PATH, SVHN_PATH, MVTEC_PATH, IMAGENETC_PATH, ADAPTIVE_PATH, mvtec_labels
import torchvision.transforms.functional as F
import requests
from PIL import Image
from tqdm import tqdm
import random

tansform_224 = transforms.Compose([
                                    transforms.Resize(224),
                                    transforms.ToTensor()
                                ])

tansform_224_gray = transforms.Compose([
                                    transforms.Resize(224),
                                    transforms.Grayscale(num_output_channels=3),
                                    transforms.ToTensor()
                                ])

tansform_32 = transforms.Compose([
                                    transforms.Resize(32),
                                    transforms.ToTensor()
                                ])

tansform_32_gray = transforms.Compose([
                                    transforms.Resize(32),
                                    transforms.Grayscale(num_output_channels=3),
                                    transforms.ToTensor()
                                ])



class GeneralDataset(torch.utils.data.Dataset):
    def __init__(self, normal_data, exposure_data, transform=None):
        self.transform = transform
        normal_data = [F.to_pil_image(x).convert('RGB') for x in normal_data]
        exposure_data = [F.to_pil_image(x).convert('RGB') for x in exposure_data]
        self.data = normal_data + exposure_data
        self.targets = [0] * len(normal_data) + [1] * len(exposure_data)

    def __getitem__(self, index):
        image = self.data[index]
        target = self.targets[index]
        
        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.data)
    
def get_dataloader(normal_dataset:str, normal_class_indx:int, exposure_dataset:str, batch_size, imagenetc=False):

    transform = None

    is_big: bool = True if normal_dataset in ['mvtec', 'ctscan'] else False
    is_colorful: bool = True if normal_dataset in ['cifar10', 'cifar100', 'svhn', 'mvtec'] else False

    if is_big:
        if is_colorful:
            transform = tansform_224
        else:
            transform = tansform_224_gray
    else:
        if is_colorful:
            transform = tansform_32
        else:
            transform = tansform_32_gray

    normal_data, testset = get_normal_class(dataset=normal_dataset, normal_class_indx=normal_class_indx, transform=transform)
    exposure_data = get_exposure(dataset=exposure_dataset, normal_dataset=normal_dataset, normal_class_indx=normal_class_indx, count=len(normal_data))

    if imagenetc:
        exposure_data += get_exposure(dataset='imagenetc', normal_dataset=normal_dataset, normal_class_indx=normal_class_indx, count=len(normal_data))

    trainset = GeneralDataset(normal_data=normal_data, exposure_data=exposure_data, transform=transform)
    del exposure_data, normal_data

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader
    


####################
#  Normal Datastes #
####################

def get_normal_class(dataset='cifar10', normal_class_indx = 0,  transform=None):

    if dataset == 'cifar10':
        return get_CIFAR10_normal(normal_class_indx, transform)
    elif dataset == 'cifar100':
        return get_CIFAR100_normal(normal_class_indx, transform)
    elif dataset == 'mnist':
        return get_MNIST_normal(normal_class_indx, transform)
    elif dataset == 'fashion':
        return get_FASHION_MNIST_normal(normal_class_indx, transform)
    elif dataset == 'svhn':
        return get_SVHN_normal(normal_class_indx, transform)
    elif dataset == 'mvtec':
        return get_MVTEC_normal(normal_class_indx, transform)
    else:
        raise Exception("Dataset is not supported yet. ")


def get_CIFAR10_normal(normal_class_indx:int, transform):
    trainset = CIFAR10(root=CIFAR10_PATH, train=True, download=True)
    trainset.data = trainset.data[np.array(trainset.targets) == normal_class_indx]

    testset = CIFAR10(root=CIFAR10_PATH, train=False, download=True, transform=transform)
    testset.targets  = [int(t!=normal_class_indx) for t in testset.targets]

    return [F.to_tensor(np.array(x).astype(np.uint8)) for x  in trainset.data], testset


def get_CIFAR100_normal(normal_class_indx:int, transform):
    trainset = CIFAR100(root=CIFAR100_PATH, train=True, download=True)
    trainset.targets = sparse2coarse(trainset.targets)
    trainset.data = trainset.data[np.array(trainset.targets) == normal_class_indx]

    testset = CIFAR100(root=CIFAR100_PATH, train=False, download=True, transform=transform)
    testset.targets = sparse2coarse(testset.targets)
    testset.targets  = [int(t!=normal_class_indx) for t in testset.targets]

    return [F.to_tensor(np.array(x).astype(np.uint8)) for x  in trainset.data], testset


def get_MNIST_normal(normal_class_indx:int, transform):
    trainset = MNIST(root=MNIST_PATH, train=True, download=True)
    trainset.data = trainset.data[np.array(trainset.targets) == normal_class_indx]

    testset = MNIST(root=MNIST_PATH, train=False, download=True, transform=transform)
    testset.targets  = [int(t!=normal_class_indx) for t in testset.targets]

    return [F.to_tensor(np.array(x).astype(np.uint8)) for x  in trainset.data], testset


def get_FASHION_MNIST_normal(normal_class_indx:int, transform):
    trainset = FashionMNIST(root=FMNIST_PATH, train=True, download=True)
    trainset.data = trainset.data[np.array(trainset.targets) == normal_class_indx]

    testset = FashionMNIST(root=FMNIST_PATH, train=False, download=True, transform=transform)
    testset.targets  = [int(t!=normal_class_indx) for t in testset.targets]

    return [F.to_tensor(np.array(x).astype(np.uint8)) for x  in trainset.data], testset

def get_SVHN_normal(normal_class_indx:int, transform):
    trainset = SVHN(root=SVHN_PATH, split='train', download=True)
    trainset.data = trainset.data[np.array(trainset.labels) == normal_class_indx].transpose(0, 2, 3, 1)

    testset = SVHN(root=SVHN_PATH, split='test', download=True, transform=transform)
    testset.labels  = [int(t!=normal_class_indx) for t in testset.labels]

    return [F.to_tensor(np.array(x).astype(np.uint8)) for x  in trainset.data], testset


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, transform=None, target_transform=None, train=True, normal=True, download=False):
        self.transform = transform

        # Check if dataset directory exists
        dataset_dir = os.path.join(root, "mvtec_anomaly_detection")
        if not os.path.exists(dataset_dir):
            if download:
                self.download_dataset(root)
            else:
                raise ValueError("Dataset not found. Please set download=True to download the dataset.")
            
        if train:
            self.data = glob(
                os.path.join(dataset_dir, category, "train", "good", "*.png")
            )

        else:
          image_files = glob(os.path.join(dataset_dir, category, "test", "*", "*.png"))
          normal_image_files = glob(os.path.join(dataset_dir, category, "test", "good", "*.png"))
          anomaly_image_files = list(set(image_files) - set(normal_image_files))
          self.data = image_files

        self.data.sort(key=lambda y: y.lower())
        self.data = [Image.open(x).convert('RGB') for x in self.data]
        self.train = train

    def __getitem__(self, index):
        image_file = self.data[index]
        image = image_file
        if self.transform is not None:
            image = self.transform(image_file)

        if os.path.dirname(image_file).endswith("good"):
            target = 0
        else:
            target = 1

        return image, target

    def __len__(self):
        return len(self.data)
    
    def download_dataset(self, root):
        url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz"
        dataset_dir = os.path.join(root, "mvtec_anomaly_detection")

        # Create directory for dataset
        os.makedirs(dataset_dir, exist_ok=True)

        # Download and extract dataset
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(os.path.join(root, "mvtec_anomaly_detection.tar.xz"), 'wb') as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
        progress_bar.close()

        os.system(f"tar -xf {os.path.join(root, 'mvtec_anomaly_detection.tar.xz')} -C {dataset_dir}")


def get_MVTEC_normal(normal_class_indx, transform):
    normal_class = mvtec_labels[normal_class_indx]

    trainset = MVTecDataset(MVTEC_PATH, normal_class, train=True, download=True)
    testset = MVTecDataset(MVTEC_PATH, normal_class, train=False, download=True, transform=transform)

    return  [F.to_tensor(np.array(x).astype(np.uint8)) for x  in trainset.data], testset


######################
#  Exposure Datastes #
######################

def get_exposure(dataset:str='cifar10', normal_dataset:str='cifar100', normal_class_indx:int = 0, count:int = 0):
    assert count > 0

    if dataset == 'cifar10':
        return get_CIFAR10_exposure(normal_dataset, normal_class_indx, count)
    elif dataset == 'cifar100':
        return get_CIFAR100_exposure(normal_dataset, normal_class_indx, count)
    elif dataset == 'mnist':
        return get_MNIST_exposure(normal_dataset, normal_class_indx, count)
    elif dataset == 'fashion':
        return get_FASHION_MNIST_exposure(normal_dataset, normal_class_indx, count)
    elif dataset == 'svhn':
        return get_SVHN_exposure(normal_dataset, normal_class_indx, count)
    elif dataset == 'mvtec':
        return get_MVTEC_exposure(normal_dataset, normal_class_indx, count)
    elif dataset == 'adaptive':
        return get_ADAPTIVE_exposure(normal_dataset, normal_class_indx, count)
    elif dataset == 'imagenetc':
        return get_IMAGENETC_exposure(count)
    else:
        raise Exception("Dataset is not supported yet. ")
    

def copy_dataset(dataset , target_count:int):
    while target_count > len(dataset):
        dataset = torch.cat((dataset, dataset.data), 0)

    return dataset


def get_CIFAR10_exposure(normal_dataset:str, normal_class_indx:int, count:int):
    exposure_train = CIFAR10(root=CIFAR10_PATH, train=True, download=True)
    exposure_test = CIFAR10(root=CIFAR10_PATH, train=False, download=True)

    if normal_dataset.lower() == 'cifar10':
        exposure_train.data = exposure_train.data[np.array(exposure_train.targets) != normal_class_indx]
        exposure_test.data = exposure_test.data[np.array(exposure_test.targets) != normal_class_indx]

    exposure_data = torch.tensor(exposure_train.data)
    del exposure_train

    if exposure_data.size(0) < count:
        exposure_data = torch.cat((exposure_data, torch.tensor(exposure_test.data)), 0)
    
    del exposure_test
    
    if exposure_data.size(0) < count:
        copy_dataset(exposure_data, count)

    indices = torch.randperm(exposure_data.size(0))[:count]
    exposure_data =  exposure_data[indices]

    return [F.to_tensor(np.array(x).astype(np.uint8)) for x  in exposure_data]

def get_CIFAR100_exposure(normal_dataset:str, normal_class_indx:int, count:int):
    exposure_train = CIFAR100(root=CIFAR100_PATH, train=True, download=True)
    exposure_train.targets = sparse2coarse(exposure_train.targets)
    exposure_test = CIFAR100(root=CIFAR100_PATH, train=False, download=True)
    exposure_test.targets = sparse2coarse(exposure_test.targets)

    if normal_dataset.lower() == 'cifar100':
        exposure_train.data = exposure_train.data[np.array(exposure_train.targets) != normal_class_indx]
        exposure_test.data = exposure_test.data[np.array(exposure_test.targets) != normal_class_indx]

    exposure_data = torch.tensor(exposure_train.data)
    del exposure_train

    if exposure_data.size(0) < count:
        exposure_data = torch.cat((exposure_data, torch.tensor(exposure_test.data)), 0)
    
    del exposure_test
    
    if exposure_data.size(0) < count:
        copy_dataset(exposure_data, count)

    indices = torch.randperm(exposure_data.size(0))[:count]
    exposure_data =  exposure_data[indices]


    return [F.to_tensor(np.array(x).astype(np.uint8)) for x  in exposure_data]



def get_MNIST_exposure(normal_dataset:str, normal_class_indx:int, count:int):    
    exposure_train = MNIST(root=MNIST_PATH, train=True, download=True)
    exposure_test = MNIST(root=MNIST_PATH, train=False, download=True)

    if normal_dataset.lower() == 'mnist':
        exposure_train.data = exposure_train.data[np.array(exposure_train.targets) != normal_class_indx]
        exposure_test.data = exposure_test.data[np.array(exposure_test.targets) != normal_class_indx]

    exposure_data = torch.tensor(exposure_train.data)
    del exposure_train

    if exposure_data.size(0) < count:
        exposure_data = torch.cat((exposure_data, torch.tensor(exposure_test.data)), 0)
    
    del exposure_test
    
    if exposure_data.size(0) < count:
        copy_dataset(exposure_data, count)

    indices = torch.randperm(exposure_data.size(0))[:count]
    exposure_data =  exposure_data[indices]

    return [F.to_tensor(np.array(x).astype(np.uint8)) for x  in exposure_data]


def get_FASHION_MNIST_exposure(normal_dataset:str, normal_class_indx:int, count:int):    
    exposure_train = FashionMNIST(root=FMNIST_PATH, train=True, download=True)
    exposure_test = FashionMNIST(root=FMNIST_PATH, train=False, download=True)

    if normal_dataset.lower() == 'fmnist':
        exposure_train.data = exposure_train.data[np.array(exposure_train.targets) != normal_class_indx]
        exposure_test.data = exposure_test.data[np.array(exposure_test.targets) != normal_class_indx]

    exposure_data = torch.tensor(exposure_train.data)
    del exposure_train

    if exposure_data.size(0) < count:
        exposure_data = torch.cat((exposure_data, torch.tensor(exposure_test.data)), 0)
    
    del exposure_test
    
    if exposure_data.size(0) < count:
        copy_dataset(exposure_data, count)

    indices = torch.randperm(exposure_data.size(0))[:count]
    exposure_data =  exposure_data[indices]

    return [F.to_tensor(np.array(x).astype(np.uint8)) for x  in exposure_data]


def get_SVHN_exposure(normal_dataset:str, normal_class_indx:int, count:int):    
    exposure_train = SVHN(root=SVHN_PATH, split='train', download=True)
    exposure_test = SVHN(root=SVHN_PATH, split='test', download=True)
    
    if normal_dataset.lower() == 'svhn':
        exposure_train.data = exposure_train.data[np.array(exposure_train.labels) != normal_class_indx]
        exposure_test.data = exposure_test.data[np.array(exposure_test.labels) != normal_class_indx]

    exposure_data = torch.tensor(exposure_train.data)
    del exposure_train

    if exposure_data.size(0) < count:
        exposure_data = torch.cat((exposure_data, torch.tensor(exposure_test.data)), 0)
    
    del exposure_test
    
    if exposure_data.size(0) < count:
        copy_dataset(exposure_data, count)

    indices = torch.randperm(exposure_data.size(0))[:count]
    exposure_data =  exposure_data[indices]

    return [F.to_tensor(np.array(x).astype(np.uint8).transpose(1, 2, 0)) for x  in exposure_data]


def get_ADAPTIVE_exposure(normal_dataset:str, normal_class_indx:int,count:int):
    exposure_data = []
    try:
        exposure_path = glob(os.path.join(ADAPTIVE_PATH, normal_dataset, f'{normal_class_indx}', "*.npy"), recursive=True)
        for path in exposure_path:
            exposure_data += np.load(path).tolist()
    except:
        raise ValueError('Wrong Exposure Address!')
        exit()

    exposure_data = torch.tensor(((torch.tensor(exposure_data) + 1)*127.5).round().clamp(0, 255), dtype=torch.uint8)


    if exposure_data.size(0) < count:
        copy_dataset(exposure_data, count)

    indices = torch.randperm(exposure_data.size(0))[:count]
    exposure_data =  exposure_data[indices]

    return exposure_data


class MVTecDatasetExposure(torch.utils.data.Dataset):
    def __init__(self, root, category=None, transform=None, download=False):
        self.transform = transform
        dataset_dir = os.path.join(root, "mvtec_anomaly_detection")
    
        if not os.path.exists(dataset_dir):
            if download:
                self.download_dataset(root)
            else:
                raise ValueError("Dataset not found. Please set download=True to download the dataset.")


        self.data = glob(os.path.join(dataset_dir, "**", "*.png"), recursive=True)

        if category is not None:
          class_files = glob(os.path.join(root, category, "**", "*.png"), recursive=True)
          self.data = list(set(self.data) - set(class_files))

        self.data.sort(key=lambda y: y.lower())
        self.data = np.array([np.array(Image.open(x).convert('RGB')) for x in self.data])

    def download_dataset(self, root):
        url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz"
        dataset_dir = os.path.join(root, "mvtec_anomaly_detection")

        # Create directory for dataset
        os.makedirs(dataset_dir, exist_ok=True)

        # Download and extract dataset
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(os.path.join(root, "mvtec_anomaly_detection.tar.xz"), 'wb') as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
        progress_bar.close()

        os.system(f"tar -xf {os.path.join(root, 'mvtec_anomaly_detection.tar.xz')} -C {dataset_dir}")


def get_MVTEC_exposure(normal_dataset:str, normal_class_indx:int, count:int):    
    exposure_data = torch.tensor(MVTecDatasetExposure(root=MVTEC_PATH, category=None if normal_dataset!='mvtec' else mvtec_labels[normal_class_indx], download=True).data)

    if exposure_data.size(0) < count:
        copy_dataset(exposure_data, count)

    indices = torch.randperm(exposure_data.size(0))[:count]
    exposure_data =  exposure_data[indices]

    return [F.to_tensor(np.array(x).astype(np.uint8)) for x  in exposure_data]

class ImageNetExposure(Dataset):
    def __init__(self, root, count, transform=None):
        self.transform = transform
        image_files = glob(os.path.join(root, "*", "*.JPEG"))
        random.shuffle(image_files)
        final_length = min(len(image_files), count)
        self.image_files = image_files[:final_length]

        self.image_files.sort(key=lambda y: y.lower())
        self.data = np.array([np.array(Image.open(x).convert('RGB')) for x in self.image_files])

def get_IMAGENETC_exposure(count:int):    
    exposure_data = torch.tensor(ImageNetExposure(IMAGENETC_PATH, count).data)

    if exposure_data.size(0) < count:
        copy_dataset(exposure_data, count)

    indices = torch.randperm(exposure_data.size(0))[:count]
    exposure_data =  exposure_data[indices]

    return [F.to_tensor(np.array(x).astype(np.uint8)) for x  in exposure_data]