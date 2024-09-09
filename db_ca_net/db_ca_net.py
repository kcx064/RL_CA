#!/usr/bin/env python3
# coding=utf-8

import torch
import torch.nn as nn

'''定义神经网络'''
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # self.flatten = nn.Flatten()
        self.hidden1 = nn.Linear(6,32)
        self.hidden2 = nn.Linear(32,64)
        self.hidden3 = nn.Linear(64,128)
        self.hidden4 = nn.Linear(128,128)
        self.hidden5 = nn.Linear(128,64)
        self.hidden6 = nn.Linear(64,32)
        self.out = nn.Linear(32,5)
    def forward(self,x):
        # x = self.flatten(x)
        x = self.hidden1(x)
        x = torch.relu(x)
        x = self.hidden2(x)
        # x = torch.sigmoid(x)
        x = torch.relu(x)
        x = self.hidden3(x)
        x = torch.relu(x)
        x = self.hidden4(x)
        x = torch.relu(x)
        x = self.hidden5(x)
        x = torch.relu(x)
        x = self.hidden6(x)
        x = torch.relu(x)
        x = self.out(x)
        return x