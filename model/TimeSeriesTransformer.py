#%%
"""
synthetic data 사용할 경우 (linear, nonlinear)
dim_T = 2
dim_C = 7
dim_O = 4
wandb_config['data'] = 'synthetic'

real data 사용할 경우 (sbk_ad)
dim_T = 2
dim_C = 4
dim_O = 4
wandb_config['data'] = 'AD'
"""
#%%
import sys
from synthetic_data.timeseries_dataset import TimeSeriesDataset
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.nn import MSELoss
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
import torch.optim as optim

import torch
torch.set_default_dtype(torch.float64)
from transformers import TimeSeriesTransformerConfig, InformerConfig
from transformers import TimeSeriesTransformerForPrediction, InformerForPrediction
import wandb
wandb.login(key="aa1e46306130e6f8863bbad2d35c96d0a62a4ddd")

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything

seed_everything(100)



class TimeSeriesTransformer(LightningModule):
    def __init__(self, config, batch_size = 32, window = 30, stride = 1, lr = 0.0005, slot_attention = True, transfer_learning = False, pretrained_path = None, model = 'tsformer'):
        super().__init__()
        self.save_hyperparameters()
        self.config = config 
        self.batch_size = batch_size
        self.window = window
        self.stride = stride
        self.lr = lr
        self.transfer_learning = transfer_learning
        self.slot_attention = slot_attention
        if pretrained_path:
            self.load_from_checkpoint(pretrained_path, config = self.config)
        else:
            if model == 'tsformer':
                self.model = TimeSeriesTransformerForPrediction(self.config)
            elif model == "informer":
                self.model = InformerForPrediction(self.config)
            self.setup_decoder(model = model)
            self.setup_last_decoder()
        if self.transfer_learning:
            self.setup_adapters()
            # self.linear = nn.Linear(dim_T + dim_C-1 + dim_O, self.config.input_size)
        self.loss_fn = nn.MSELoss()
        self.masks = None

    def forward(self, batch):
        batch['prev_timefeatures'] = batch['prev_timefeatures'].unsqueeze(-1) if batch['prev_timefeatures'].ndim <3 else batch['prev_timefeatures']
        batch['curr_timefeatures'] = batch['curr_timefeatures'].unsqueeze(-1) if batch['curr_timefeatures'].ndim <3 else batch['curr_timefeatures']
        past_values = torch.cat([#batch['prev_treatments'], 
                                    batch['prev_covariates'], 
                                    batch['prev_outcomes']], dim = -1)
        # past_time_features = torch.stack([batch['prev_timefeatures']] * (self.config.num_time_features + self.config.num_dynamic_real_features), dim =2)
        past_time_features = torch.cat([batch['prev_timefeatures'],
                                        batch['prev_treatments']], dim = -1)
        past_observed_mask = batch['prev_mask'][:,:,2:]
        future_values = torch.cat([#batch['curr_treatments'], 
                                    batch['curr_covariates'], 
                                    batch['curr_outcomes']], dim = -1)
        # future_time_features = torch.stack([batch['curr_timefeatures']] * (self.config.num_time_features + self.config.num_dynamic_real_features), dim =2)
        future_time_features = torch.cat([batch['curr_timefeatures'],
                                        batch['curr_treatments']], dim = -1)
        future_observed_mask = batch['curr_mask'][:,:,2:]
        if self.transfer_learning:
            past_values = self.input_adapter(past_values)
            past_observed_mask = self.input_adapter(past_observed_mask)
            future_values = self.input_adapter(future_values)
            future_time_features = self.input_adapter(future_time_features)
            future_observed_mask = self.input_adapter(future_observed_mask)
        outputs = self.model(past_values = past_values, 
                        past_time_features = past_time_features, 
                        past_observed_mask = past_observed_mask, 
                        future_values = future_values, 
                        future_time_features = future_time_features,
                        future_observed_mask = future_observed_mask)  # You might need to adjust this call based on actual API
        # print(outputs.encoder_last_hidden_state.shape) # 64, 82, 64
        if self.slot_attention:
            outputs = outputs.encoder_last_hidden_state.unsqueeze(2).repeat(1, 1, self.config.input_size,1) #(b, w', f , d_model)
            outputs = outputs.reshape(outputs.shape[0], self.config.input_size, -1) # (b, f, w' * d_model)
            outputs = self.decoder(outputs) # b, f, w * f
            masks = outputs.reshape(outputs.shape[0], self.config.input_size, self.config.prediction_length, -1, 2) # b, f, w, f, 2
            slots, masks = outputs.split([1, 1],dim = -1)
            slots = slots.squeeze(-1) # b, f, w, f
            masks = masks.squeeze(-1) # b,f,w,f
            masks = nn.Softmax(dim = 1)(masks) #b,f,w,f
            # slots = past_values.permute(0,2,1).unsqueeze(-1) # b,f,w,1
            outputs = torch.sum(slots * masks, dim = 1) # b, w, f
        else: 
            masks = None
        return outputs, masks
                                       
    def setup_decoder(self, model = None):
        input_dim = self.config.context_length * self.config.d_model
        if model == "informer":
            input_dim = input_dim // 2
        self.decoder = nn.Sequential(
            nn.Linear(input_dim,self.config.prediction_length * self.config.d_model),
            nn.LayerNorm(self.config.prediction_length * self.config.d_model),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(self.config.prediction_length * self.config.d_model, self.config.prediction_length * self.config.d_model),
            nn.LayerNorm(self.config.prediction_length * self.config.d_model),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(self.config.prediction_length * self.config.d_model, 
                    self.config.prediction_length * self.config.input_size * 2))

    def setup_adapters(self):
        input_feature_size = self.config.input_size
        self.input_adapter = nn.Linear(input_feature_size, 9)
        self.output_adapter = nn.Linear(self.config.prediction_length * self.config.d_model, 
                                        self.config.prediction_length * (self.config.input_size + 1))
        self.decoder[-1] =self.output_adapter

    def configure_optimizers(self):
        lr = self.lr
        optimizer = AdamW(self.parameters(), lr = lr)
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma = 0.99)

        return [optimizer], [lr_scheduler]
    def training_step(self, batch, batch_idx):
        outputs, _ = self(batch)
        future_values = torch.cat([#batch['curr_treatments'], 
                            batch['curr_covariates'], 
                            batch['curr_outcomes']], dim = -1)
        
        if self.slot_attention:
            loss = self.loss_fn(outputs, future_values)
        else:
            loss = outputs.loss
        if self.slot_attention:
            loss = self.loss_fn(outputs, future_values)
        else:
            loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs, self.masks = self(batch)
        future_values = torch.cat([#batch['curr_treatments'], 
                            batch['curr_covariates'], 
                            batch['curr_outcomes']], dim = -1)

        if self.slot_attention:
            loss = self.loss_fn(outputs, future_values)
        else:
            loss = outputs.loss
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        outputs, _ = self(batch)
        future_values = torch.cat([#batch['curr_treatments'], 
                            batch['curr_covariates'], 
                            batch['curr_outcomes']], dim = -1)

        if self.slot_attention:
            loss = self.loss_fn(outputs, future_values)
        else:
            loss = outputs.loss
        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def forecasting(self, batch):
        # self.model.eval()
        past_values = torch.cat([#batch['prev_treatments'], 
                                batch['prev_covariates'], 
                                batch['prev_outcomes']], dim = -1)
        # past_time_features = torch.stack([batch['prev_timefeatures']] * (self.config.num_time_features + self.config.num_dynamic_real_features), dim =2)
        past_time_features = torch.cat([batch['prev_timefeatures'],
                                        batch['prev_treatments']], dim = -1)

        past_observed_mask = batch['prev_mask'][:,:,2:]
        if self.transfer_learning:
            past_values = self.linear(past_values)
            past_observed_mask = self.linear(past_observed_mask)
        future_time_features = torch.stack([batch['curr_timefeatures']] * (self.config.num_time_features + self.config.num_dynamic_real_features), dim =2)

        forecasts = self.model.generate(
            past_values = past_values,
            past_time_features = past_time_features,
            past_observed_mask = past_observed_mask,
            future_time_features = future_time_features,
        )
        forecasts = forecasts.sequences.numpy()
        return forecasts, batch['prev_raw']
    
    def _reconstruct_timeseries(self, batched_data):
        self.total_length = (self.batch_size - 1) * self.stride + self.window - 1
        reconstructed = torch.zeros(self.total_length)
        # Fill the reconstructed tensor
        for i in range(self.batch_size):
            start = i * self.stride
            reconstructed[start:start + self.window - 1] = batched_data[i]
        return reconstructed

    
    def visualize(self, batch, batch_idx, feature, scaling_params, all_unscaled, timestamp):
        forecasts, actual_unscaled = self.forecasting(batch)
        feature_list = list(all_unscaled.columns)
        feature_idx = feature_list.index(feature)
        
        val_timestamp = timestamp[train_size:].reset_index(drop = True)
        actual = self._reconstruct_timeseries(actual_unscaled[:,:,feature_idx])
        forecasts_unscaled = forecasts[:,:,:,feature_idx] * scaling_params['stds'][feature_idx] \
            + scaling_params['means'][feature_idx]
        predict = self._reconstruct_timeseries(torch.tensor(forecasts_unscaled.mean(axis =1)))
        std = self._reconstruct_timeseries(torch.tensor(forecasts_unscaled.std(axis =1)))
        
        reconstructed_timestamp = val_timestamp[batch_idx*self.total_length : batch_idx*self.total_length + self.total_length]
        

        plt.figure(figsize=(15, 5))
        plt.plot(reconstructed_timestamp, actual, label='Actual Data', color='blue', marker = 'o')
        plt.plot(reconstructed_timestamp, predict, label='Forecasted Data', color='red', linestyle='--')
        plt.fill_between(reconstructed_timestamp, predict - std, predict + std, alpha = 0.2, label = "+/- 1-std")
        plt.title(f'Forecast vs Actuals for {feature}')
        plt.xlabel('Time Steps')
        plt.ylabel('Values')
        plt.legend()
        plt.show()

if __name__ == "__main__":

    data_type = 'nonlinear'
    FILE_PATH = '/Users/sujinchoi/Library/CloudStorage/OneDrive-postech.ac.kr/ADMetalearning/데이터/nonlinear_causal.npy'
    data = np.load(FILE_PATH)

    # Assuming `data` is your dataset as a NumPy array or a PyTorch tensor of shape (4237, 10)
    data_tensor = torch.tensor(data) if isinstance(data, np.ndarray) else data
    # data_tensor = data_tensor[:,[0,1,2,3,6,10,19,20,21,22,23,
    #                              24,25,26,27,30,34,43,44,45,46,
    #                              47,48,49,50,53,57,66,67,68,69]]

    # Split data
    # dim_T = 1
    # dim_C = 6
    # dim_O = 4
    dim_T = 2
    dim_C = 7
    dim_O = 4
    dim_time = 1
    # dim_T = 2
    # dim_C = 2
    # dim_O = 4
    # dim_time = 3
    dim_input = dim_C+ dim_O
    wandb_config = {
        'data_type': data_type,
        'data': 'synthetic', # 'AD'
        'window': 45,
        'stride': 1,
        'batch_size': 32,
        'epoch': 25,
        'lr': 0.0005,
        'loss': 'gelu',
        'scheduler': 'ExponentialLR',
        'model': 'tsformer'


    }
    window = 45
    stride = 1
    batch_size = 32
    epoch = 25
    # Define the split ratio
    train_ratio = 0.5
    val_ratio = 0.3
    test_ratio = 0.2

    # Calculate the number of samples in the training set
    train_size = int(data_tensor.size(0) * train_ratio)
    val_size = int(data_tensor.size(0) * val_ratio)
    train_data_tensor = data_tensor[:train_size]
    val_data_tensor = data_tensor[train_size:train_size + val_size]
    test_data_tensor = data_tensor[train_size + val_size:]
    train_data = TimeSeriesDataset(train_data_tensor, dim_T = dim_T, dim_C = dim_C, dim_O = dim_O, 
                                    window_size = wandb_config["window"], stride = wandb_config["stride"], data_type = wandb_config['data'])
    train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
    val_data = TimeSeriesDataset(val_data_tensor, dim_T = dim_T, dim_C = dim_C, dim_O = dim_O, 
                                    window_size = wandb_config["window"], stride = wandb_config["stride"], data_type = wandb_config['data'])
    val_dataloader = DataLoader(val_data, batch_size = batch_size, shuffle = False)
    test_data = TimeSeriesDataset(test_data_tensor, dim_T = dim_T, dim_C = dim_C, dim_O = dim_O, 
                                    window_size = wandb_config["window"], stride = wandb_config["stride"], data_type = wandb_config['data'])
    test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle = False)


    config = TimeSeriesTransformerConfig(
        d_model=64,  # Size of the embeddings (and the hidden state)
        encoder_layers=2,  # Number of encoder layers
        decoder_layers=2,  # Number of decoder layers
        encoder_attention_heads=2,  # Number of attention heads in the encoder
        decoder_attention_heads=2,  # Number of attention heads in the decoder
        dropout=0.1,  # Dropout rate
        activation_function='gelu',  # Activation function
        prediction_length=window - 1,  # Predicting one step ahead
        input_size = dim_input, 
        context_length=window -1 -7,  # How many steps the encoder looks back
        num_static_categorical_features=0,  # No static categorical features
        num_static_real_features=0,  # No static real features
        num_dynamic_real_features=dim_T,  # All features are dynamic
        num_time_features=dim_time,  # No additional time features
        num_parallel_samples = 10,
        output_hidde_states = True 
    )
    # config = PatchTSTconfig(
    #     num_input_channels = dim_input,
    #     context_length = window -1 -7,
    #     num_hidden_layers = 2,
    #     d_model = 64,
    #     prediction_length = window - 1,
    #     num_parallel_samples = 10,
    #     mask_type = 'forecast'
    # )
    # config = InformerConfig(
    #     d_model=64,  # Size of the embeddings (and the hidden state)
    #     encoder_layers=2,  # Number of encoder layers
    #     decoder_layers=2,  # Number of decoder layers
    #     encoder_attention_heads=2,  # Number of attention heads in the encoder
    #     decoder_attention_heads=2,  # Number of attention heads in the decode r
    #     dropout=0.1,  # Dropout rate
    #     activation_function='gelu',  # Activation function
    #     prediction_length=window - 1,  # Predicting one step ahead
    #     input_size = dim_input, 
    #     context_length=window -1 -7,  # How many steps the encoder looks back
    #     num_static_categorical_features=0,  # No static categorical features
    #     num_static_real_features=0,  # No static real features
    #     num_dynamic_real_features=dim_input,  # All features are dynamic
    #     num_time_features=1,  # No additional time features
    #     num_parallel_samples = 10,
    #     output_hidde_states = True 
    # )

    model = TimeSeriesTransformer(config, batch_size = wandb_config["batch_size"], 
                                  window = wandb_config["window"], 
                                  stride = wandb_config["stride"], 
                                  lr = wandb_config["lr"],
                                  slot_attention=True,
                                  transfer_learning = False,
                                  model = wandb_config['model']) 
                                #   pretrained_path= "/Users/sujinchoi/Library/CloudStorage/OneDrive-postech.ac.kr/ADMetalearning/코드/sbk_bestmodel.ckpt")
    # model = TimeSeriesTransformer.load_from_checkpoint("/Users/sujinchoi/Library/CloudStorage/OneDrive-postech.ac.kr/ADMetalearning/코드/sbk_bestmodel.ckpt")
    wandb_logger = WandbLogger(project = f'TimeSeriesTransformer_{data_type}', name = f'{data_type}-{epoch}_{window}_{batch_size}')
    wandb.config.update(wandb_config)

    trainer = pl.Trainer(max_epochs = wandb_config['epoch'], logger = wandb_logger)
    trainer.fit(model = model, train_dataloaders= train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, test_dataloader)
    trainer.save_checkpoint(f'{data_type}-{epoch}_{window}_{batch_size}.ckpt')



# %%



