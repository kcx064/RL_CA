#!/usr/bin/env python3
# coding=utf-8


from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from ca_net import NeuralNetwork


matrix = np.array([
[1.0000, 1.0000, 1.0000, 1.0000], 
[-0.0884, 0.0884, 0.0884, -0.0884], 
[0.0884, -0.0884, 0.0884, -0.0884], 
[0.0166 ,0.0166, -0.0166, -0.0166]], dtype=np.float32)


'''
主函数
'''
if __name__ ==('__main__'):

    ca_net = NeuralNetwork()
    ca_net = torch.load('./ca.pt')
    ca_net = ca_net.to('cpu')
    print(ca_net)

    virtual_control = np.array([[15], [0], [0], [0]], dtype = np.float32)
    # virtual_control = np.array([15, 0, 0, 0], dtype = np.float32)
    
    virtual_control = torch.from_numpy(virtual_control)
    virtual_control = virtual_control.to('cpu').t()
    print(virtual_control)

    control_value = ca_net(virtual_control)

    print(control_value)