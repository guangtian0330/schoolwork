# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim

"""
Created on Thu Nov 30 17:04:46 2023

@author: Guangtian Gong, Sisong Li, Zijian Han
"""

"""
Develop and train 1D CNN (Convolutional Neural Network) models appropriate for
sequential sensor data classification.

TBD by Sisong
"""


"""
Develop and train GRU (Gated Recurrent Unit) networks for sequential sensor
data classification.
"""
class GRUClassifier(nn.Module) :
    def __init__(self, input_size, hidden_size, output_size) :
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first = True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x) :
        _, h_n = self.gru(x)
        output = self.fc(h_n[-1])
        return output

if __name__ == '__main__':
    output_size = 3
    hidden_size = 128
    input_size = 64
    
    model = GRUClassifier(input_size, hidden_size, output_size)

