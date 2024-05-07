#%%
import sys
sys.path.append('/home/user126/TFformer')
from pytorch_lightning import LightningModule
import torch
torch.set_default_dtype(torch.float64)
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage
import torch.optim as optim
import math
import numpy as np
import pdb
import logging
from torch.utils.data import DataLoader
from model.TFformer_transformer import TransformerEncoderBlock, PositionalEncoding
from model.utils import VariableSelection, SeriesDecomposition
import wandb 
from pytorch_lightning.loggers import WandbLogger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#%%
class TFformer(LightningModule):

    def __init__(self,
                 dim_treatments, dim_covariates, dim_outcomes, dim_statics,
                 seq_hidden_units, iw, ow, num_heads, dropout_rate, num_layer):
        super().__init__()
        self.save_hyperparameters()
        
        # params for input 
        self.dim_T = dim_treatments
        self.dim_C = dim_covariates
        self.dim_O = dim_outcomes
        self.dim_S = dim_statics

        self.is_decomposition = True
        if self.is_decomposition:
            self.decomposition = SeriesDecomposition(kernel_size= 7)
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
        #                                                       nhead=self.num_heads, dropout = self.dropout_rate, batch_first = True)
        
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers =self.num_layer)
        self.positional_encoding = PositionalEncoding(d_model = self.seq_hidden_units*self.total, max_len=iw)
        self.transformer_encoder = nn.ModuleList([self.basic_block_cls(
            hidden = self.seq_hidden_units, attn_heads = self.num_heads, head_size = self.head_size,
            feed_forward_hidden = self.seq_hidden_units, dropout = self.dropout_rate,
            self_positional_encoding_k=None, self_positional_encoding_v=None
            ) for _ in range(self.num_layer)])
        
        output_unit = 1
        self.linear1 = nn.Sequential(
            nn.Linear(self.seq_hidden_units, self.seq_hidden_units//2),
            nn.ReLU(),
            nn.Linear(self.seq_hidden_units//2, output_unit)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(iw, (iw+ow)//2),
            nn.ReLU(),
            nn.Linear((iw+ow)//2, ow)
        )

        # dropout
        self.output_dropout = nn.Dropout(self.dropout_rate)

        # loss function
        self.loss_fn = nn.MSELoss()
        self.causal_mask = self._generate_causal_mask(iw, self.total)
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

        # hidden = self.transformer_encoder(embedded, self.causal_mask)
        for block in self.transformer_encoder:
            hidden = block(embedded, self.causal_mask) # (batch_size, seq_len, seq_hidden_units)
        # hidden = hidden.reshape(batch_size, timesteps, feature_size, -1).permute(0,2,1,3)
        
        # feed only outcome

        output_dim = outcomes.size(-1)
        # hidden = hidden[:, feature_size - output_dim: ]
        # hidden = hidden[:,-1]
        
        pred = self.linear1(hidden)
        pred = pred.reshape(batch_size, timesteps, -1)
        pred = self.linear2(pred.permute(0,2,1)).permute(0,2,1)
        # pred = pred.view(hidden.shape[0], self.timestep, -1)
        return pred
    
    def _generate_causal_mask(self, timestep, dim_total):
        causal_mask = torch.tril(torch.ones(timestep, timestep))
        causal_mask = causal_mask.repeat_interleave(dim_total, dim=1)
        _remainder_mask = torch.ones(timestep * (dim_total - 1), timestep * dim_total)
        causal_mask = torch.cat((causal_mask, _remainder_mask))
        return causal_mask

    def _get_actual_output(self, batch):
        curr_treatment = batch['curr_treatments']
        curr_covariates = batch['curr_covariates']
        curr_outcomes = batch['curr_outcomes']
        # if self.variable_selection:
        #     curr_treatment, _ = self.treatment_selection_layer(batch['curr_treatments'])
        # else:
        #     curr_treatment = batch['curr_treatments']
        # if self.variable_selection:
        #     curr_covariates, _ = self.covariates_selection_layer(batch['curr_covariates'])
        # else:
        #     curr_covariates = batch['curr_covariates']
        actual = torch.cat([curr_treatment, curr_covariates, curr_outcomes], dim = -1)
        # actual = batch['curr_outcomes']
        return actual

    def configure_optimizers(self):
        lr = 0.001 
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
    from synthetic_data.timeseries_dataset import TimeSeriesDataset
    # def to_float32(batch):
    #     return {k: v.to(torch.float32) for k, v in batch.items()}
    import gc
    gc.collect()
    # num_points = 365*10  # Number of time points
    # num_feature = 13    # Number of series
    # window = 12*7
    # stride = 5
    # data = create_dataset(num_feature = num_feature, num_points = num_points)
    # train_dataset= LinearDataset(data[:int(len(data)*0.5)], num_feature, window, stride)
    # train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    # val_dataset= LinearDataset(data[int(len(data)*0.5):int(len(data)*0.7)], num_feature, window, stride)
    # val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
    # test_dataset= LinearDataset(data[int(len(data)*0.7):], num_feature, window, stride)
    # test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    data = np.load('/home/user126/TFformer/synthetic_data/data/sbk_AD.npy')
    dim_T = 4
    dim_C = 4
    dim_O = 4
    epochs = 100
    batch_size = 64
    iw = 10*7
    ow = 3*7
    stride = 1
    samples = len(data)
    train_data = data[:int(samples * 0.5)]
    val_data = data[int(samples * 0.5):int(samples * 0.7)]
    test_data = data[int(samples * 0.7):]

    train_data = TimeSeriesDataset(train_data, dim_T, dim_C, dim_O, iw, ow, stride)
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)

    val_data = TimeSeriesDataset(val_data, dim_T, dim_C, dim_O, iw, ow, stride)
    val_dataloader = DataLoader(val_data, batch_size=64, shuffle=False)
    
    test_data = TimeSeriesDataset(test_data, dim_T, dim_C, dim_O, iw, ow, stride)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

    wandb.init()
    wandb.login(key="aa1e46306130e6f8863bbad2d35c96d0a62a4ddd")
    wandb_logger = WandbLogger(project = 'TFFormer', name = f'TFformer_linear_{epochs}')

    trainer = pl.Trainer(accelerator = "cpu",max_epochs = 100, log_every_n_steps = 40, logger = wandb_logger)
    model = TFformer(dim_treatments=dim_T, dim_covariates=dim_C, dim_outcomes = dim_O, dim_statics=None,
                     seq_hidden_units=32, iw = iw, ow = ow, num_heads = 2, dropout_rate=0.2, num_layer = 3)
    
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader)





# %%
