#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 15:32:20 2021

@author: ziyi
"""


import json
import torch

with open('R2R_train.json') as f:
    json_data = [json.loads(line) for line in f]
    #data = json.loads(f)
 #%%   
i = 0
q7_data = []

for path in json_data:
    if path['scan'] == '5q7pvUzZiYa':
        if path['language']=='en-US' or path['language']=='en-IN':
            q7_data.append(path)
            i += 1
with open('5q7pvUzZiYa.json','w',encoding = 'utf-8') as f:    
    json.dump(q7_data, f)
    #%%
with open('5q7pvUzZiYa.json') as f:
    json_data = [json.loads(line) for line in f]
    #data = json.loads(f)
#%%
pairs = []
for path in json_data[0]:
    ins = path['instruction']
    path_seq = path['path']
    path_tensor = []
    for vp in path_seq:
        tensor_vp = [0,0,0,0]
        for i in [1,2,3,4]:
            filename = 'image_features/' + vp + '_skybox' + str(i) + '_sami.pt'
            tensori = torch.load(filename)
            tensor_vp[i-1] = tensori.squeeze().numpy()
            #print(tensori.squeeze().numpy())
        #print(tensor_vp)
        path_tensor.append(tensor_vp)
    pair = [path_tensor, ins]
    #print(pair)
    pairs.append(pair)
#%%
import pickle
with open('pairs.txt','wb') as fp:
    pickle.dump(pairs,fp)

#%% single image input
#######################################################################################
with open('connectivity_map/5q7pvUzZiYa_connectivity.json') as f:
    connect_map = [json.loads(line) for line in f]
poseDict = {}
for vp in connect_map[0]:
    img_id = vp['image_id']
    posemat = vp['pose']
    pose = (posemat[3],posemat[7],posemat[11])
    poseDict[img_id] = pose
    path_1img = []
for path in json_data[0]:
    path_1 = []
    path_seq = path['path']
    ins = path['instruction']
    for i in range(len(path_seq)-1):
        vp1 = path_seq[i]
        vp2 = path_seq[i+1]
        pose1 = poseDict[vp1]
        pose2 = poseDict[vp2]
        x = pose2[0]-pose1[0]
        y = pose2[1]-pose1[1]
        if x > 0:
            if y > 0:
                if abs(x) > abs(y):
                    index = 4
                else:
                    index = 3
            else:
                if abs(x) > abs(y):
                    index = 4
                else:
                    index = 1
        else:
            if y > 0:
                if abs(x) > abs(y):
                    index = 2
                else:
                    index = 3
            else:
                if abs(x) > abs(y):
                    index = 2
                else:
                    index = 1
                    
        filename = vp1 + '_skybox' + str(index) + '_sami.pt'
        path_1.append(filename)
    path_1img.append([path_1,ins])
    
print(path_1img)
                
            
    
    
    
    
    
    
    
    