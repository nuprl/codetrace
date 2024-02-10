"""
Idea: given a prompt where LLM predicts <FILL> incorrectly, train a model CAUSAL_MASKER
top produce a MASK (matrix) that, when applied to LLM components, will produce the correct prediction.

Training data for CAUSAL_MASKER
input: LLM, MASK over components, prompt, incorrect prediction, target prediction
output: MASK over components

Architecture:

CAUSAL_MASKER takes a matrix and produces a new matrix
it has a loss that is computed:

    Loss= LLM_MASK(prompt)[logit_of_incorrect_prediction] - LLM_MASK(prompt)[logit_of_correct_prediction]
    
the goal is to minimize this loss, so that LLM_MASK(prompt) produces the correct prediction

IDEA: use a custom Convolutional Network

SIMPLE VERSION: 1 token prediction

TODO: training loop
model takes input mask, the output mask is then passed to next train item
loss is calculated over batch logits (each same mask thankfully)
mask needs to be applied in a batch 

Modle inp-out are the mask
Loss is derived from a function that uses mask applied to traiuning items
"""

"""
example of a mask for a model with 2 layers, 2 attention heads and an MLP dim of size 6
Note: mlp dim needs to be divisible by attn heads

dimensions are (l, a, (d//a)+1)

TODO: disconnected computation graph somewhere

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import sys
from typing import List
import random
import numpy as np
import datasets

  
class MatrixAutoEncoder(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.conv1 = nn.Conv1d(24, 16, 3, padding=1)  
        self.pool = nn.MaxPool1d(2, 2)
        self.t_conv2 = nn.ConvTranspose1d(16, 24, 2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.sigmoid(self.t_conv2(x))
        return x

def z_score_normalize(matrix):
    mean = matrix.mean()
    std_dev = matrix.std()
    normalized_matrix = (matrix - mean) / std_dev
    return normalized_matrix

def train_loop():
    """
    model not updating?
    """
    batch_size = 5
    model = MatrixAutoEncoder()
    model.train()

    train_data = np.random.randint(0, 2, (batch_size, 24, 16)).astype(np.float32)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=1)

    optimizer = torch.optim.Adam(model.parameters(),  lr=0.1)
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Apply gradient clipping


    print("Training...")
    # number of epochs to train the model
    n_epochs = 10
            
    criterion = nn.BCELoss()
    for epoch in range(1, n_epochs+1):
        # monitor training loss
        train_loss = 0.0
        
        for data in train_loader:
            optimizer.zero_grad()

            output= model(data)

            
            with torch.enable_grad():
                loss = criterion(output, data)
                loss.backward()
            

            optimizer.step()
            print(f"Batch loss: {loss.item()}")
            train_loss += loss.item()
            
            
        # print avg training statistics 
        train_loss = train_loss/len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch, 
            train_loss
            ))



if __name__ == "__main__":

    train_loop()
