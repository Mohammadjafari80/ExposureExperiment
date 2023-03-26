import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
from data.constants import ADAPTIVE_PATH
import torchvision.transforms.functional as F
import requests
from PIL import Image
from tqdm import tqdm

def getAdaptiveExposureDataset(normal_dataset, normal_class_indx):
    class AdaptiveExposureDataset(torch.utils.data.Dataset):
        def __init__(self, root, transform=None, target_transform=None, train=True, normal=True, download=False):
            self.transform = transform
            self.data = []
            try:
                file_paths = glob(os.path.join(ADAPTIVE_PATH, normal_dataset, f'{normal_class_indx}', "*.npy"), recursive=True)
                for path in file_paths:
                    self.data += list(np.load(path))
                new_data = []
                for x in self.data:
                    image_file = torch.from_numpy(x)
                    scaled = ((image_file + 1)*127.5).round().clamp(0,255).to(torch.uint8).cpu()
                    new_data.append(scaled/255)
                self.data = new_data    
                
            except:
                raise ValueError('Wrong Exposure Address!')
            self.train = train

        def __getitem__(self, index):
            image_file = self.data[index]
            
            image = image_file
            if self.transform is not None:
                image = self.transform(image_file)

            target = 0

            return image, target

        def __len__(self):
            return len(self.data)
    return AdaptiveExposureDataset
