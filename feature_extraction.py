#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 16:23:39 2021

@author: ziyi
"""


import torch
from torchvision import models
import torch.nn as nn

model = models.alexnet(pretrained=True)

class AlexNetConv(nn.Module):
    def __init__(self):
        super(AlexNetConv,self).__init__()
        self.features = nn.Sequencial(
            
            *list(model.features.children())[:-3]
            )
    def forward(self,x):
        x = self.features(x)
        return x
    
from PIL import  Image

filename = '0e84cf4dec784bc28b78a80bee35c550_i0_0.jpg'
from torchvision import transforms
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')
    
with torch.no_grad():
    output = model(input_batch)

print(output)
#%%
import os
from os import walk


for (dirpath, dirnames, filenames) in walk(os.getcwd()):
    
    filenames = filenames

for filename in filenames:
    if filename[-4:] == '.jpg':
        input_image = Image.open(filename)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)
        with torch.no_grad():
            output = model(input_batch)
        tensor_name = filename[:-3] +'pt'
        torch.save(output,tensor_name)
        
    
