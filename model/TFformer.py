#%%
import sys
sys.path.append('/Users/sujinchoi/Desktop/TFformer')

from pytorch_lightning import LightningModule
import torch
# torch.set_default_dtype(torch.float64)
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage
import torch.optim as optim
import math
import numpy as np
import pdb
import logging
from torch.utils.data import DataLoader, Dataset
from model.TFformer_transformer import TransformerEncoderBlock

import wandb 
from pytorch_lightning.loggers import WandbLogger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GatedLinearUnit(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Linear(input_dim, output_dim)
    
    def forward(self, inputs):
        return self.linear(inputs) * torch.sigmoid(self.sigmoid(inputs))
    
class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, self.input_dim)
        self.fc2 = nn.Linear(self.input_dim, self.input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.gated_linear_unit = GatedLinearUnit(self.input_dim, self.output_dim)
        self.layer_norm = nn.LayerNorm(self.output_dim)
        self.project = None  # Initialized only if needed in the forward method

    def forward(self, inputs, context_vector = None):
        if context_vector is not None:
            x = torch.cat((inputs, context_vector), dim = -1)
            x = F.elu(self.fc1(x))
        else:
            x = F.elu(self.fc1(inputs))
        x = self.fc2(x)
        x = self.dropout(x)
        if inputs.shape[-1] != self.output_dim:
            if self.project is None:
                self.project = nn.Linear(self.input_dim, self.output_dim)
            inputs = self.project(inputs)
        x = inputs + self.gated_linear_unit(x)
        x = self.layer_norm(x)
        if torch.isnan(x).any():
            print(f"NaN detected in GRN output")
        return x

class VariableSelection(nn.Module):
    def __init__(self, feature, num_variable, output_unit, dropout_rate = 0.2, context_size = 0):
        super().__init__()
        self.feature = 'prev_' + str(feature)
        self.grns = nn.ModuleList([GatedResidualNetwork(output_unit, output_unit, dropout_rate) for _ in range(num_variable)])
        self.grn_concat = GatedResidualNetwork(num_variable * output_unit + context_size, num_variable, dropout_rate)
        self.softmax = nn.Softmax(dim = -1)
        self.embedding = nn.Linear(1, output_unit)

    def forward(self, batch, context_vector = None):
        '''
        Params:
            feature: treatments/covariates/outcomes
        '''
        batch_size, timesteps, num_features = batch.shape
        inputs = [] 
        for i in range(num_features):
            inputs.append(self.embedding(batch[:,:,i].unsqueeze(-1)))
        v = torch.cat(inputs, dim=-1) # (32, 99, 35)
        v = self.grn_concat(v) # (32, 99, 7)
        v = self.softmax(v)
        sparse_weights = v.unsqueeze(-2) # (32, 99, 7, 1)
        # v shape (batch, seq_len, num_feat, 1)
        x = torch.stack([grn(input_tensor) for grn, input_tensor in zip(self.grns, inputs)], dim=-1) # (32, 99, 5, 7)

        combined = sparse_weights * x
        temporal_ctx = torch.sum(combined, dim = -1) # (32, 99, 5)
        return temporal_ctx, sparse_weights # (batch_size, seq_length, output_unit)

'''
class VariableSelectionNetwork(nn.Module):
    def __init__(self, dim_T, dim_C, dim_O):
        super().__init__()
        self.t = 2
        self.c = 5
        self.o = 1
        self.out_dim = (self.t + self.c + self.o)
        self.treatment_layer = nn.Linear(dim_T,self.t)
        self.pos_treatment = PositionalEncoding(self.t)

        self.covariates_layer = nn.Linear(dim_C,self.c)
        self.pos_covariates = PositionalEncoding(self.c)
        
        # self.outcome_layer = nn.Linear(dim_O,o)
        # self.pos_outcome = PositionalEncoding(o)
    def forward(self, batch):
        # x: (batch, timestep, features)
        selected_treatments = F.relu(self.treatment_layer(batch['prev_treatments']))
        selected_treatments = self.pos_treatment(selected_treatments)
        selected_covariates = F.relu(self.treatment_layer(batch['prev_covariates']))
        selected_covariates = self.pos_covariates(selected_covariates)
        # selected_outputs = F.relu(self.treatment_layer(batch['outcomes']))
        outputs = self.pos_outcome(batch['prev_outcomes'])
        return torch.cat([selected_treatments, selected_covariates, outputs], dim = 2)'''
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1]) if div_term.size(0) > 1 \
                else torch.cos(position * torch.exp(torch.tensor(-math.log(10000.0) / d_model)))
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:,:seq_len, :]
        return self.dropout(x)
    
class TFformer(LightningModule):

    def __init__(self,
                 dim_treatments, dim_covariates, dim_outcomes, dim_statics,
                 seq_hidden_units, num_heads, dropout_rate, num_layer, timestep):
        super().__init__()
        self.save_hyperparameters()
        
        # params for input 
        self.dim_T = dim_treatments
        self.dim_C = dim_covariates
        self.dim_O = dim_outcomes
        self.dim_S = dim_statics
        self.timestep = timestep

        self.variable_selection = False
        if self.variable_selection:
            self.treatment_selection_layer = VariableSelection(feature = 'treatments', num_variable=self.dim_T, output_unit=2)
            self.covariates_selection_layer = VariableSelection(feature = 'covariates', num_variable=self.dim_C, output_unit=5)
            self.dim_T = 2
            self.dim_C = 5
        self.total = self.dim_T + self.dim_C + self.dim_O

        self.embedding_layer = nn.Linear(1, seq_hidden_units)
        self.basic_block_cls = TransformerEncoderBlock

        # params for basic block cls
        self.seq_hidden_units = seq_hidden_units
        self.num_heads = num_heads
        self.head_size = seq_hidden_units // num_heads
        self.dropout_rate = dropout_rate
        self.num_layer = num_layer

        self.ema = True
        self.beta = 0.99
        if self.ema:
            self.ema_treatment = ExponentialMovingAverage(self.parameters(), decay = self.beta)

        # transformer blocks 
        # encoder_layer = nn.TransformerEncoderLayer(d_model=self.seq_hidden_units, 
        #                                                       nhead=self.num_heads, dropout = self.dropout_rate)
        
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers =self.num_layer)
        self.positional_encoding = PositionalEncoding(d_model = self.seq_hidden_units*self.total)
        self.transformer_encoder = nn.ModuleList([self.basic_block_cls(
            hidden = self.seq_hidden_units, attn_heads = self.num_heads, head_size = self.head_size,
            feed_forward_hidden = self.seq_hidden_units, dropout = self.dropout_rate,
            self_positional_encoding_k=None, self_positional_encoding_v=None
            ) for _ in range(self.num_layer)])
        

        self.predict_outcomes = nn.Sequential(
            nn.Linear(self.seq_hidden_units, self.seq_hidden_units),
            nn.ReLU(),
            nn.Linear(self.seq_hidden_units, 1)
        )

        # dropout
        self.output_dropout = nn.Dropout(self.dropout_rate)

        # loss function
        self.loss_fn = nn.MSELoss()
        self.causal_mask = self._generate_causal_mask(timestep, self.total)
        self.register_buffer('tfcausal_mask', self.causal_mask)

    def forward(self,batch):
        # x (batch_size, timesteps, feature_size)
        treatment = batch['prev_treatments']
        covariates = batch['prev_covariates']
        outcomes = batch['prev_outcomes']
        if self.variable_selection:
            treatment, w_T = self.treatment_selection_layer(batch[self.treatment_selection_layer.feature])            
            covariates, w_C = self.covariates_selection_layer(batch[self.covariates_selection_layer.feature])
        batch = torch.cat([treatment, covariates, outcomes], dim = -1)
        batch_size, timesteps, feature_size = batch.shape
                        
        flattened = torch.flatten(batch, 1).unsqueeze(-1) # (batch_size, timesteps * feature_size, 1)
        embedded = self.embedding_layer(flattened) # (batch_size, timesteps * feature_size, seq_hidden_units)
        if self.positional_encoding is not None:
            embedded = embedded.view(batch_size, timesteps, -1)
            embedded = self.positional_encoding(embedded)
            embedded = embedded.view(batch_size, timesteps * feature_size, -1)


        # embedded = embedded.permute(1,0,2) 
        torch.save(embedded, 'embedded.pt') # (batch_size, seq_len, seq_hidden_units)

        # hidden = self.transformer_encoder(embedded, self.causal_mask)
        for block in self.transformer_encoder:
            hidden = block(embedded, self.causal_mask) # (batch_size, seq_len, seq_hidden_units)
        
        # feed only outcome
        output_dim = outcomes.size(-1)
        indices = [list(range(i * feature_size - output_dim, i * feature_size)) for i in range(1, timesteps + 1)]
        indices = [idx for sublist in indices for idx in sublist]  # Flatten the list of lists
        outcome_hidden = hidden[:, indices, :]
        
        pred = self.predict_outcomes(outcome_hidden)
        pred = pred.view(hidden.shape[0], self.timestep, -1)
        return pred
    
    def _generate_causal_mask(self, timestep, dim_total):
        causal_mask = torch.tril(torch.ones(timestep, timestep))
        causal_mask = causal_mask.repeat_interleave(dim_total, dim=1)
        _remainder_mask = torch.ones(timestep * (dim_total - 1), timestep * dim_total)
        causal_mask = torch.cat((causal_mask, _remainder_mask))
        return causal_mask

    def _get_actual_output(self, batch):
        # if self.variable_selection:
        #     curr_treatment, _ = self.treatment_selection_layer(batch['curr_treatments'])
        # else:
        #     curr_treatment = batch['curr_treatments']
        # if self.variable_selection:
        #     curr_covariates, _ = self.covariates_selection_layer(batch['curr_covariates'])
        # else:
        #     curr_covariates = batch['curr_covariates']
        # actual = torch.cat([curr_treatment, curr_covariates, batch['curr_outcomes']], dim = -1)
        actual = batch['curr_outcomes']
        return actual

    def configure_optimizers(self):
        lr = 0.01 
        optimizer = optim.Adam(self.parameters(), lr = lr)
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma = 0.99)

        return [optimizer], [lr_scheduler]
    
    def training_step(self, batch, trainer=True):
        pred = self(batch)
        actual = self._get_actual_output(batch)
        if self.ema:
            with self.ema_treatment.average_parameters():
                pred = self(batch)
        
        mse_loss = self.loss_fn(pred, actual)
        print("mse_loss", mse_loss)
        # mse_loss = (batch['active_entries'] * mse_loss).sum() / batch['active_entries'].sum()
        if trainer:
            self.log(f'train_mse_loss', mse_loss, on_epoch = True, on_step = False, sync_dist = True)
        return mse_loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        actual = self._get_actual_output(batch)
        if self.ema:
            with self.ema_treatment.average_parameters():
                pred = self(batch)

        mse_loss = self.loss_fn(pred, actual)
        self.log(f'validation_mse_loss', mse_loss.mean(), on_epoch=True, on_step=False, sync_dist=True)
    

    def test_step(self, batch, batch_idx):
        pred = self(batch)
        actual = self._get_actual_output(batch)
        if self.ema:
            with self.ema_treatment.average_parameters():
                pred = self(batch)
        mse_loss = self.loss_fn(pred, actual)
        self.log(f'test_mse_loss', mse_loss.mean(), on_epoch=True, on_step=False, sync_dist=True)
    

if __name__ == "__main__":
    from synthetic_data.linear_causal import LinearDataset
    num_points = 1000  # Number of time points
    num_series = 13    # Number of series
    window = 100
    stride = 5
    dim_T = 2
    dim_C = 7
    dim_O = 4
    epochs = 10
    batch_size = 64
    # def to_float32(batch):
    #     return {k: v.to(torch.float32) for k, v in batch.items()}

    train_dataset= LinearDataset(num_points, num_series, window, stride)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_dataset= LinearDataset(num_points//4, num_series, window, stride)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
    test_dataset= LinearDataset(num_points//4, num_series, window, stride)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    wandb.login(key="aa1e46306130e6f8863bbad2d35c96d0a62a4ddd")
    wandb_logger = WandbLogger(project = 'TFFormer', name = f'TFformer_linear_{epochs}')

    trainer = pl.Trainer(accelerator = "cpu",max_epochs = 20, log_every_n_steps = 40, logger = wandb_logger)
    model = TFformer(dim_treatments=dim_T, dim_covariates=dim_C, dim_outcomes = dim_O, dim_statics=None,
                     seq_hidden_units=10, num_heads = 2, dropout_rate=0.2, num_layer = 3, timestep= window -1)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)





# %%
