from pytorch_lightning import LightningModule
import torch
from torch import nn
import torch.optim as optim
import numpy as np
import pdb
import logging
from torch.utils.data import DataLoader, Dataset
from model.utils_transformer import TransformerMultiInputBlock

import wandb 

class TFformer(LightningModule):

    def __init__(self,
                 dim_treatments, dim_covariates, dim_outcomes, dim_statics,
                 seq_hidden_units, num_heads, dropout_rate, num_layer):
        super().__init__()
        self.basic_block_cls = TransformerMultiInputBlock
        
        # params for input 
        self.dim_T = dim_treatments
        self.dim_C = dim_covariates
        self.dim_O = dim_outcomes
        self.dim_S = dim_statics

        # params for basic block cls
        self.seq_hidden_units = seq_hidden_units
        self.num_heads = num_heads
        self.head_size = seq_hidden_units // num_heads
        self.dropout_rate = dropout_rate
        self.num_layer = num_layer

        # transformer blocks 
        self.transformer_blocks = nn.ModuleList([self.basic_block_cls(self.seq_hidden_units, 
                                                                      self.num_heads, self.head_size, 
                                                                      self.seq_hidden_units * 4,self.dropout_rate,
                                                                      self.dropout_rate ,
                                                                      self_positional_encoding_k=self.self_positional_encoding_k,
                                                                      self_positional_encoding_v=self.self_positional_encoding_v
                                                                      ,n_inputs=self.n_inputs,disable_cross_attention = False,
                                                                      isolate_subnetwork='_') for _ in range(self.num_layer)])


        # dropout
        self.output_dropout = nn.Dropout(self.dropout_rate)



    def forward(self,
                ):
        
        
        pass
    
    def build_br(self,):
        pass
    
    def gated_residual_network(self,):
        
        pass
        
    def variable_selection(self,):
        pass

    def training_step(self, batch, batch_idx, trainer=True):

        return mse_loss

    def validation_step(self, batch, batch_idx):
        if self.ema:
            with self.ema_treatment.average_parameters():
                _, outcome_pred = self(batch) 
        else:
            _, outcome_pred = self(batch) 
        mse_loss = nn.functional.mse_loss(outcome_pred, batch['outputs'], reduce=False)
        self.log(f'validation_mse_loss', mse_loss.mean(), on_epoch=True, on_step=False, sync_dist=True)
    

    def test_step(self, batch, batch_idx):
        if self.ema:
            with self.ema_treatment.average_parameters():
                _, outcome_pred = self(batch) 
        else:
            _, outcome_pred = self(batch) 
        mse_loss = nn.functional.mse_loss(outcome_pred, batch['outputs'], reduce=False)
        self.log(f'test_mse_loss', mse_loss.mean(), on_epoch=True, on_step=False, sync_dist=True)
    

    def configure_optimizers(self):
        lr = 0.01 
        optimizer = optim.Adam(self.parameters(), lr = lr)
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma = 0.99)

        return [optimizer], [lr_scheduler]
