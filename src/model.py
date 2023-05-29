
import numpy as np
import torch
from torch import nn, Tensor
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import torch.utils.data as data_utils
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from torch import optim 
from tqdm import tqdm 
import numpy as np
import pandas as pd
import scipy.io
import math
import os
import ntpath
import sys
import logging
import time
import sys

from torch.autograd import Variable
from einops.layers.torch import Rearrange


import copy
 

 

class MLP_block(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            # nn.Dropout(dropout),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class MixerBlock(nn.Module):
    def __init__(self, token, channel, hidden_size):
        super(MixerBlock, self).__init__()
         
        self.token_mixer = nn.Sequential(
            nn.LayerNorm(token),
            Rearrange('b n d -> b d n'),
            MLP_block(input_size=channel, hidden_size=hidden_size),
            Rearrange('b n d -> b d n')
        )
        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(token),
            MLP_block(input_size=token, hidden_size=hidden_size)
        )

        self.a1 = nn.Parameter(torch.ones(token))  
        self.a2 = nn.Parameter(torch.ones(token)) 
        
         
    def forward(self, x):
        x = x +   self.a1 * self.token_mixer(x)
        x = x +   self.a2 * self.channel_mixer(x)
        return x


class MixerMLP(nn.Module):
    def __init__(self, token, channel, hidden_size, depth=1):
        super(MixerMLP, self).__init__()
        self.depth = depth
        self.layer = nn.ModuleList()
        for _ in range(depth):
            layer = MixerBlock(token, channel, hidden_size)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        for block in self.layer:
            x = block(x)
        return x
 
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class MoEPred(nn.Module):
    def __init__(self, d_in, n_experts=32):
        super().__init__()
         
        self.n_experts = n_experts
        self.gating = nn.ModuleList([nn.Linear(d_in, 1, bias=False) for i in range(self.n_experts)])
        self.experts = nn.ModuleList(
            [
            nn.Sequential(
            nn.Linear(d_in, 16),
            nn.GELU(), 
            nn.Linear(16, 1),
            )
            for i in range(self.n_experts)
            ]
        )

        

    def forward(self, x):
        
        # subx = torch.chunk(x, self.n_experts, 1)
        output_of_experts = []
        gating_score_of_experts = []
        for expert_id in range(self.n_experts):
            gating_score_of_experts.append(self.gating[expert_id](x))
            expert_out =self.experts[expert_id](x)   
            output_of_experts.append(expert_out)

        output_of_experts = torch.stack(output_of_experts, 2)  # (batch_size, d_in, n_experts)
        gating_score_of_experts = torch.stack(gating_score_of_experts, 1)  # (batch_size, n_experts, 1)
        moe_out = torch.bmm(output_of_experts, gating_score_of_experts.softmax(1))
        xi = moe_out.squeeze(1) 

        return xi

 
 

class Net(nn.Module):
    def __init__(self, window_size,  hidden_dim=16):
        super(Net, self).__init__()
         
        self.gru_encoder =   nn.GRU(1, hidden_dim//2, batch_first=True, bidirectional=True, num_layers= 2)
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, 2, batch_first=True, dropout=0.2)
        self.mixer = MixerMLP(hidden_dim, window_size, hidden_dim//4, 2)
        self.moe_pred = MoEPred(hidden_dim)
         
         
        
    def forwad_once(self, x ):
        
        x = rearrange(x, 'b n t -> b t n') 
        x, _ = self.gru_encoder(x)
         
        x, _ = self.multihead_attn(x,x,x)
         
        x = self.mixer(x).contiguous()
        
        # x = x.view(x.size(0), -1)

         
         
        x= reduce(x, 'b t c -> b c', 'mean')
        return x 
    
   
    def forward(self, x): 
       
        x = self.forwad_once(x)
        pred = self.moe_pred(x)
        return pred 
    
 

 
 