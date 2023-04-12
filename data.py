import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision.transforms.functional import adjust_contrast, adjust_sharpness



class DataLoader:
    
    def __init__(self, config):
        
        self.path = config['path']
        self.imgs, self.targets, self.labels = self.load_data()
    
    
    def targets(self):
        return self.targets
    
    
    def classes(self):
        return self.labels
    
    
    def data(self):
        return self.imgs
    
    
    def load_data(self):
    
        imgs, targets, labels= [], [], []
        
        for target, file in enumerate(os.listdir(self.path)):
            class_name = re.split("[_.]", file)[-2]
            labels.append(class_name)
            
            file = os.path.join(self.path, file)
            img_samples = np.load(file)
            
            imgs.extend(img_samples)
            target_samples = [target] * img_samples.shape[0]
            targets.extends(target_samples)
        
        imgs = torch.tensor(imgs)
        targets = torch.LongTensor(targets)
                
        return  imgs, targets, labels



class QuickDrawDataset(Dataset):
    
    def __init__(self, config):
        super(QuickDrawDataset, self).__init__()
        self.config = config
        self.path = config['path']
        self.transforms = config['transforms']
    

class QucikDrawImageDataset(QuickDrawDataset):
    
    def __init__(self, loader, config):
        super(QucikDrawImageDataset, self).__init__(config)
        self.loader = loader
        self.img_shape = 
        
    def __len__(self):
        return len(self.loader.targets())
    
    
    def __getitem__(self, index):
        
        img = self.loader.data()[index].reshape(self.img_shape)
        target = self.loader.targets()[index]
        
        if self.transforms:
            img = self.transforms(img)
        
        img = adjust_contrast(img, contrast_factor=2)
        img = adjust_sharpness(img, sharpness_factor=2)
        
        
        return img, target
        






class QuickDrawStrokeDataset(QuickDrawDataset):
    
    def __init__(self,
                 transform=None,
                 target_transform=None):
        
        super(QuickDrawDataset, self).__init__()
  
        
        
        