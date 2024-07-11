#!/usr/bin/env python3
# coding=utf-8

import torch
import torch.nn as nn

'''定义神经网络'''
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.hidden1 = nn.Linear(4,128)
        self.hidden2 = nn.Linear(128,128)
        self.hidden3 = nn.Linear(128,64)
        self.out = nn.Linear(64,4)
    def forward(self,x):
        x = self.flatten(x)
        x = self.hidden1(x)
        x = torch.relu(x)
        x = self.hidden2(x)
        x = torch.sigmoid(x)
        x = self.hidden3(x)
        x = torch.relu(x)
        x = self.out(x)
        return x