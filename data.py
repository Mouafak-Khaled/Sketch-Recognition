import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision.transforms.functional import adjust_contrast, adjust_sharpness
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor



class QuickDrawDataset(Dataset):
    
    def __init__(self, config, transforms=None):
        super(QuickDrawDataset, self).__init__()
        self.config = config
        self.path = config['path']
        self.transforms = transforms
        
    
    def classes(self):
        
        classes = []
        for file in os.lisdir(self.path):
            class_name = re.split("[_.]", file)[-2]
            classes.append(class_name)
        
        return classes

        
    
class QucikDrawImageDataset(QuickDrawDataset):
    
    def __init__(self, config, transforms=None):
        super(QucikDrawImageDataset, self).__init__(config, transforms=transforms)
        self.img_shape = config['img_shape']
        self.img_files = os.listdir(self.path)
        
        
    def __len__(self):
        return sum([np.load(os.path.join(self.path, file)).
                    shape[0] for file in self.img_files])

    
    def __getitem__(self, index):
        
        if index < 0:
            index += self.__len__()
        
        for target, file in enumerate(self.img_files):
            imgs = np.load(os.path.join(self.path, file))
            if index < len(imgs):
                img = imgs[index].reshape(self.img_shape)
                
                if self.transforms:
                    img = self.transforms(img)
                    
                img = adjust_contrast(img, contrast_factor=2)
                img = adjust_sharpness(img, sharpness_factor=2)
                
                return img, target
            
            index -= len(imgs)
        
        
        
class QuickDrawStrokeDataset(QuickDrawDataset):
    
    def __init__(self,
                 transform=None,
                 target_transform=None):
        
        super(QuickDrawDataset, self).__init__()
  
        
        
        