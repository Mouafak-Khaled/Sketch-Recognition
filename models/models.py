import os
import torch
import torch.nn as nn

class ClassifierFactory:
    
    def create_classfier(self, type , config):
        
        if type.lower() == 'cnn':
            return SketchRecognitionCNN(config)
        
        # elif type.lower() == 'rnn':
        #     return SketchRecognitionRNN()
        
        else:
            return ValueError("Invalid Classification type!")
        

class SketchRecognitionCNN(nn.Module):
    
    def __init__(self, config):
        
        super(SketchRecognitionCNN, self).__init__()
        
        layers_config = config['layers']
        self.model = self._build_layers(layers_config)
        self.model.apply(self._init_weights)
        
    
    def _build_layers(self, layers_config):
        
        layers = []
        for config in layers_config:
            
            layer_type = config['type']
            layer_args = config['args']
            layer = getattr(nn, layer_type)(**layer_args)
            layers.append(layer)
        
        return nn.Sequential(*layers)
            
            
    def _init_weights(self, m):
        
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.1)  
    
    
    def forward(self, x):
        
        return self.model(x)  
        
        
        
        