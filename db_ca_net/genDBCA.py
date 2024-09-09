#!/usr/bin/env python3
# coding=utf-8

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from tqdm import trange, tqdm

from db_ca_net import NeuralNetwork

import math

data_len = 100000

matrix = np.array([
[1.0000, 1.0000, 1.0000, 1.0000], 
[-0.0884, 0.0884, 0.0884, -0.0884], 
[0.0884, -0.0884, 0.0884, -0.0884], 
[0.0166 ,0.0166, -0.0166, -0.0166]], dtype=np.float32)

def ca_forward(ca_in):
    L_1 = 0.898
    L_2 = 0.383
    L_3 = 0.825
    c = 0.0389
    Psi = 0.8381

    T_1 = ca_in[0]
    T_2 = ca_in[1]
    T_3 = ca_in[2]
    delta_1 = ca_in[3]
    delta_2 = ca_in[4]

    matrix_db = np.array([[math.sin(delta_1)*math.sin(Psi), -math.sin(delta_2)*math.sin(Psi), 0],
                          [math.sin(delta_1)*math.cos(Psi),  math.sin(delta_2)*math.cos(Psi), 0],
                          [-math.cos(delta_1),              -math.cos(delta_2),               -1],
                          [L_1*math.cos(delta_1),           -L_1*math.cos(delta_2),            0],
                          [L_2*math.cos(delta_1),            L_2*math.cos(delta_2),         -L_3],
                          [L_2*math.sin(delta_1)*math.cos(Psi)+L_1*math.sin(delta_1)*math.sin(Psi) + c, L_2*math.sin(delta_2)*math.cos(Psi)+L_1*math.sin(delta_2)*math.sin(Psi)-c, c]
                          ],dtype=np.float32)

    return matrix_db @ np.array([[T_1],[T_2],[T_3]])



class Mydataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        # self.idx = list()
        # for item in x:
        #     self.idx.append(item)
        # pass

    def __getitem__(self, index):
        input_data = self.x[:,index]
        target = self.y[:,index]
        return input_data, target

    def __len__(self):
        num_rows_shape, num_cols_shape = self.x.shape
        return num_cols_shape



 
'''定义训练函数'''
# def train(dataloader, model, loss_fn, optimizer):
#     model.train()
#     pbar = tqdm(dataloader, desc="Training")
#     # 记录优化次数
#     # num=1
#     for X,y in pbar:
#         X,y = X.to(device),y.to(device)
#         # 自动初始化权值w
#         pred = model.forward(X)
#         loss = loss_fn(pred,y) # 计算损失值
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         loss_value = loss.item()
#         pbar.set_postfix({'loss': f'{loss_value:.8f}'})

def train(dataloader, model, loss_fn, optimizer, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        iter_n = 1
        loss_value_mean = 0

        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            pred = model(X)  # 也可以使用 model.forward(X)，但通常直接调用 model(X) 更简洁
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            # cal mean loss
            loss_value_mean = iter_n/(iter_n+1)*loss_value_mean + 1/((iter_n+1))*loss_value
            iter_n += 1

            pbar.set_postfix({'mean loss': f'{loss_value_mean:.8f}'})
'''
主函数
'''
if __name__ ==('__main__'):

    # 生成n组4x1的向量，每个元素在0到8.5之间
    '''
    生成数据的输入时我们训练模型的输出，生成数据的输出是训练模型的输入
    '''
    my_input = np.random.uniform(low=0, high=1000, size=(5, data_len)).astype(np.float32)
    my_target = np.zeros((6, data_len), dtype=np.float32)

    for i in range(data_len):
        # my_target[:,i] = matrix @ my_input[:,i]
        my_target[:,i] = ca_forward(my_input[:,i]).squeeze()
        # print('output(v):',my_target[:,i], 'input(T):',my_input[:,i])

    '''
    第一个参数是训练模型的输入，第二个是训练模型的输出
    '''
    datasets = Mydataset(my_target, my_input)

    # train_dataloader = DataLoader(datasets) 
    train_dataloader = DataLoader(datasets, batch_size=4, num_workers=1) 

    # for i, (input_data, target) in enumerate(train_dataloader):
    #     print('')
    #     print('input_data%d' % i, input_data)
    #     print('target%d' % i, target)

     
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    print('Training device:', device)

    model = NeuralNetwork().to(device)
    # model = torch.load('./ca_db.pt')
    print(model)
    
    '''
    建立损失函数和优化算法
    '''
    #交叉熵损失函数
    # loss_fn = nn.CrossEntropyLoss()
    
    # 创建一个均方差损失函数对象  
    loss_fn = nn.MSELoss() 

    # 优化算法为随机梯度算法/ Adam 优化算法
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    
    train(train_dataloader, model, loss_fn, optimizer)

    torch.save(model,'./ca_db.pt')

    


