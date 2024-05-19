#%%
"""
# main_v1.py: new model (use Timeseries transformer decoder)
# real, synthetic, treatment 유무 parameter로 조절
# baseline (tsTFormer huggingface) slot attention False 하면 실행 가능

synthetic data 사용할 경우 (linear, nonlinear)
dim_T = 2
dim_C = 7
dim_O = 4
wandb_config['data'] = 'synthetic'

real data 사용할 경우 (sbk_ad)
dim_T = 2
dim_C = 2
dim_O = 4
wandb_config['data'] = 'AD'
"""
#%%
import sys
sys.path.append("/Users/sujinchoi/Desktop/TF_slot")
from timeseries_dataset_combine import TimeSeriesDatasetUnique
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
import torch.nn.functional as F

import csv
from sklearn.preprocessing import MinMaxScaler
import torch
torch.set_default_dtype(torch.float64)
from transformers import TimeSeriesTransformerConfig, InformerConfig
from transformers import TimeSeriesTransformerModel,TimeSeriesTransformerForPrediction, InformerForPrediction
import wandb
wandb.login(key="aa1e46306130e6f8863bbad2d35c96d0a62a4ddd")
from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning.utilities.seed import seed_everything

# seed_everything(100)
import random
import os

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
seed_everything(100)




class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.rand(1, 1, dim))

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots = None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_sigma.expand(b, n_s, -1)
        slots = torch.normal(mu, sigma)

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))

        return slots
    
    
class TimeSeriesTransformer(LightningModule):
    def __init__(self, config, batch_size = 32, window = 30, stride = 1, lr = 0.0005, 
                 dim_T=2, dim_C = 7, dim_O = 4,
                 slot_attention = True, num_slot = None,
                 treatment=False, slot_iter = 5,
                 transfer_learning = False, pretrained_path = None, 
                 model = 'tsformer', output_window_size = 10):
        super().__init__()
        self.save_hyperparameters()

        self.config = config 
        self.batch_size = batch_size
        self.window = window
        self.stride = stride
        self.lr = lr
        self.transfer_learning = transfer_learning
        self.slot_attention = slot_attention
        self.num_slot = num_slot if num_slot is not None else self.config.input_size

        self.dim_T = dim_T
        self.dim_C = dim_C
        self.dim_O = dim_O

        self.pred_len = output_window_size

        self.treatment = treatment

        if pretrained_path:
            self.load_from_checkpoint(pretrained_path, config = self.config)
        else:
            if model == 'tsformer':
                if self.slot_attention == False: # use baseline model
                    self.model = TimeSeriesTransformerForPrediction(self.config)
                else:   # Ours
                    self.model = TimeSeriesTransformerModel(self.config)
                    self.encoder = self.model.get_encoder()
                    self.decoder = self.model.get_decoder()
            elif model == "informer":
                self.model = InformerForPrediction(self.config)


            self.get_slot = SlotAttention(
                num_slots = self.num_slot,
                dim = self.config.d_model,
                iters = slot_iter,
            )

            self.fc = nn.Sequential(
                nn.Linear(self.config.d_model,self.config.d_model//2),
                nn.LayerNorm(self.config.d_model//2),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(self.config.d_model//2,self.config.input_size + 1)
            )
            
        self.loss_fn = nn.MSELoss()
        self.masks = None

    def forward(self, batch, val_flag = False):
        batch['prev_timefeatures'] = batch['prev_timefeatures'].unsqueeze(-1) if batch['prev_timefeatures'].ndim <3 else batch['prev_timefeatures']
        batch['curr_timefeatures'] = batch['curr_timefeatures'].unsqueeze(-1) if batch['curr_timefeatures'].ndim <3 else batch['curr_timefeatures']
        if self.treatment:
            past_values = torch.cat([
                                        batch['prev_treatments'], 
                                        batch['prev_covariates'], 
                                        batch['prev_outcomes']], dim = -1)
            past_time_features = torch.cat([batch['prev_timefeatures']], dim = -1)
            past_observed_mask = batch['prev_mask']
            future_values = torch.cat([
                                        batch['curr_treatments'], 
                                        batch['curr_covariates'], 
                                        batch['curr_outcomes']], dim = -1)
            future_time_features = torch.cat([batch['curr_timefeatures']], dim = -1)
            future_observed_mask = batch['curr_mask']
        
        else:
            past_values = torch.cat([batch['prev_covariates'], 
                                        batch['prev_outcomes']], dim = -1)
            past_time_features = torch.cat([batch['prev_timefeatures'],
                                        batch['prev_treatments'], 
                                            ], dim = -1)
            past_observed_mask = batch['prev_mask'][:,:,self.dim_T:]
            future_values = torch.cat([
                                        batch['curr_covariates'], 
                                        batch['curr_outcomes']], dim = -1)
            future_time_features = torch.cat([batch['curr_timefeatures'],
                                            batch['curr_treatments'], 
                                            ], dim = -1)
            future_observed_mask = batch['curr_mask'][:,:,self.dim_T:]

        if self.slot_attention:
            transformer_inputs, _, _, _ = self.model.create_network_inputs(
                past_values = past_values,
                past_time_features=past_time_features,
                past_observed_mask=past_observed_mask,
                future_values=future_values,
                future_time_features=future_time_features,
            )
            # encoder_inputs [32, 44+5, 36]
            encoder_inputs = transformer_inputs[:,:self.config.context_length]
            encoder_outputs = self.encoder(
                inputs_embeds=encoder_inputs,
            ) # [32, 44, 64]
            slots_masks = self.get_slot(encoder_outputs[0]) # b, n, d
            slots_masks = slots_masks.reshape(-1, slots_masks.shape[-1]).unsqueeze(1) # b * n, 1, d
            slots_masks = slots_masks.repeat(1,self.config.prediction_length,1) # b * n, w, d

            decoder_inputs = transformer_inputs[:,self.config.context_length:] # [32,5,36]
            decoder_inputs = decoder_inputs.unsqueeze(1).repeat(1,self.num_slot,1,1).view(-1, self.config.prediction_length, decoder_inputs.shape[-1]) #[32*numslot,5,64]
            decoder_outputs = self.decoder(
                inputs_embeds =decoder_inputs,
                encoder_hidden_states=slots_masks,
            )

            decoder_outputs = decoder_outputs.last_hidden_state #[32,5,64]
            decoder_outputs = self.fc(decoder_outputs)
            decoder_outputs = decoder_outputs.view(-1, self.num_slot, self.config.prediction_length, self.config.input_size + 1)  # Adjust dimensions as needed
            recons, masks = decoder_outputs.split([self.config.input_size, 1], dim=-1)
            masks = nn.Softmax(dim=1)(masks)
            outputs = torch.sum(recons * masks, dim=1)  # Recombine reconstructed image parts weighted by normalized masks

        else: 
            # for training
            outputs = self.model(past_values = past_values, 
                            past_time_features = past_time_features, 
                            past_observed_mask = past_observed_mask, 
                            future_values = future_values, 
                            future_time_features = future_time_features) 
            masks = None

        return outputs, masks
                                  

    def configure_optimizers(self):
        lr = self.lr
        optimizer = AdamW(self.parameters(), lr = lr)
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma = 0.99)
        return [optimizer], [lr_scheduler]
    
    def training_step(self, batch, batch_idx):
        if self.treatment:
            future_values = torch.cat([
                                batch['curr_treatments'], 
                                batch['curr_covariates'], 
                                batch['curr_outcomes']], dim = -1)
        else:
            future_values = torch.cat([
                                batch['curr_covariates'], 
                                batch['curr_outcomes']], dim = -1)

        if self.slot_attention:
            outputs, _ = self(batch)
            loss = self.loss_fn(outputs, future_values)
        else:
            outputs, _ = self(batch)
            loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        if self.treatment:
            future_values = torch.cat([
                                batch['curr_treatments'], 
                                batch['curr_covariates'], 
                                batch['curr_outcomes']], dim = -1)
        else:
            future_values = torch.cat([
                                batch['curr_covariates'], 
                                batch['curr_outcomes']], dim = -1)

        if self.slot_attention:
            outputs,masks= self(batch)
            loss = self.loss_fn(outputs, future_values)
            self.log('recursive validation loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        else:
            batch['prev_timefeatures'] = batch['prev_timefeatures'].unsqueeze(-1) if batch['prev_timefeatures'].ndim <3 else batch['prev_timefeatures']
            batch['curr_timefeatures'] = batch['curr_timefeatures'].unsqueeze(-1) if batch['curr_timefeatures'].ndim <3 else batch['curr_timefeatures']
        
            if self.treatment:
                past_values = torch.cat([batch['prev_treatments'],batch['prev_covariates'],batch['prev_outcomes']],dim = -1)
                past_time_features = torch.cat([batch['prev_timefeatures']],dim = -1)
                past_observed_mask = batch['prev_mask']
                future_time_features = torch.cat([batch['curr_timefeatures']],dim = -1)
            else:
                past_values = torch.cat([batch['prev_covariates'],batch['prev_outcomes']],dim = -1)
                past_time_features = torch.cat([batch['prev_treatments'],
                                                batch['prev_timefeatures']],dim = -1)
                past_observed_mask = batch['prev_mask'][:,:,dim_T:]
                future_time_features = torch.cat([batch['curr_treatments'],
                                                  batch['curr_timefeatures']],dim = -1)

            generated = self.model.generate(
                past_values = past_values,
                past_time_features = past_time_features,
                past_observed_mask = past_observed_mask,
                future_time_features = future_time_features,
            )
            generated = generated.sequences.mean(dim=1)
            loss = self.loss_fn(generated,future_values)
            self.log('validation loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        if self.treatment:
            future_values = torch.cat([
                                batch['curr_treatments'], 
                                batch['curr_covariates'], 
                                batch['curr_outcomes']], dim = -1)
        else:
            future_values = torch.cat([
                                batch['curr_covariates'], 
                                batch['curr_outcomes']], dim = -1)
        if self.slot_attention:
            outputs,masks= self(batch)
            loss = self.loss_fn(outputs, future_values)
            self.log('recursive test loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        else:
            batch['prev_timefeatures'] = batch['prev_timefeatures'].unsqueeze(-1) if batch['prev_timefeatures'].ndim <3 else batch['prev_timefeatures']
            batch['curr_timefeatures'] = batch['curr_timefeatures'].unsqueeze(-1) if batch['curr_timefeatures'].ndim <3 else batch['curr_timefeatures']
        
            if self.treatment:
                past_values = torch.cat([batch['prev_treatments'],batch['prev_covariates'],batch['prev_outcomes']],dim = -1)
                past_time_features = torch.cat([batch['prev_timefeatures']],dim = -1)
                past_observed_mask = batch['prev_mask']
                future_time_features = torch.cat([batch['curr_timefeatures']],dim = -1)
            else:
                past_values = torch.cat([batch['prev_covariates'],batch['prev_outcomes']],dim = -1)
                past_time_features = torch.cat([batch['prev_treatments'],
                                                batch['prev_timefeatures']],dim = -1)
                past_observed_mask = batch['prev_mask'][:,:,dim_T:]
                future_time_features = torch.cat([batch['curr_treatments'],
                                                  batch['curr_timefeatures']],dim = -1)

            generated = self.model.generate(
                past_values = past_values,
                past_time_features = past_time_features,
                past_observed_mask = past_observed_mask,
                future_time_features = future_time_features,
            )
            generated = generated.sequences.cpu().numpy()[:,-1] # or mean?
            loss = self.loss_fn(torch.tensor(generated).to(future_values.device),future_values)
            self.log('validation loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    

if __name__ == "__main__":
    # 'sbk', 'linear', 'nonlinear'
    # data_type = 'sbk'
    '''
    # data_type
        - AD, synthetic, real
    # data
        - sbk_AD, linear_causal, nonlinear_causal, weather, ETTm1, ETTm2, national_illness

    '''
    ########## parameters ############
    data_name = "sbk_AD"
    data_type = "AD"
    
    accelerator = "cpu"
    treatment = True  
    # True: predict treatment using model -> include treatment into input
    slot_attention = False  # True: use our model, False: simpel Timeseries Transformer from hugginface
    if slot_attention:
        model_name = "ours"
    else:
        model_name = "hf_tsformer"

    if treatment:
        treatment_txt = "_treat"
    else:
        treatment_txt = ""

    wandb_config = {
        'context_length': 60,
        'max_lag': 1,   # size of input value (context length)
        'pred_length': 30,
        'stride': 1,
        'batch_size': 32,
        'epoch': 30,
        'lr': 0.0005,
        'num_slot': 30,  
        'slot_iter': 5,
        'data_type': data_name,
        'data': data_type,  # "synthetic", "AD"
        'loss': 'gelu',
        'scheduler': 'ExponentialLR',
        'model': 'tsformer',
        'scaling': "mean",   # True, "mean", "std" is only meaningful values
    }
    ##################################

    file_format = "csv"
    dim_time = 1
    if data_name == "nonlinear_causal" or data_name =="linear_causal":
        dim_T = 2
        dim_C = 7
        dim_O = 4
        file_format = "npy"
        data_type = "synthetic"
    elif data_name == "sbk_AD":
        dim_T = 2
        dim_C = 2
        dim_O = 4
        dim_time = 3
        data_type = "AD"
        file_format = "npy"
    elif data_name == "weather":
        dim_T = 10
        dim_C = 6
        dim_O = 5
        sel_order = [0,3,6,8,9,10,11,12,19,20,1,2,4,5,7,13,14,15,16,17,18]
    elif data_name == "EETm1":
        dim_T = 1
        dim_C = 3
        dim_O = 3
        sel_order = [5,0,1,2,3,4,6]
    elif data_name == "EETm2":
        dim_T = 3
        dim_C = 2
        dim_O = 2
        sel_order = [0,1,3,2,4,5,6]
    elif data_name == "national_illness":
        dim_T = 2
        dim_C = 3
        dim_O = 2
        sel_order = [5,6,0,1,2,3,4]
    else:
        raise NotImplementedError
    
    FILE_PATH = f"./data/{data_name}.{file_format}"
    
    if data_type=='real':
        data = []
        with open(FILE_PATH, "r", newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                data.append(row)
        #data = np.genfromtxt(FILE_PATH, delimiter=",", skip_header=1)[:3000]
        length = 10000
        data = np.asarray(data[1:length+1])
        timefeature = data[:,0]
        dayfeature = np.tile(np.arange(7),2000)[:length].reshape(-1,1)
        monthfeature = []
        for str in timefeature:
            monthfeature.append(int(str[6]))
        monthfeature = np.asarray(monthfeature).reshape(-1,1)      
        data = data[:,1:].astype(float)
        data = data[:,sel_order]
        # scaler = MinMaxScaler()
        # data = scaler.fit_transform(data)
        data = np.concatenate((data,monthfeature),axis = 1)
    else: 
        data = np.load(FILE_PATH)

    data_tensor = torch.tensor(data) if isinstance(data, np.ndarray) else data
    
    if treatment:
        dim_input = dim_C + dim_O + dim_T
        dynamic_feat = 0
    else:
        dim_input = dim_C + dim_O
        dynamic_feat = dim_T

    context_length = wandb_config['context_length']
    stride = wandb_config['stride']
    batch_size = wandb_config['batch_size']
    epoch = wandb_config['epoch']

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

    train_data = TimeSeriesDatasetUnique(train_data_tensor, dim_T = dim_T, dim_C = dim_C, dim_O = dim_O,
                                    input_window_size = wandb_config["context_length"] + wandb_config["max_lag"], stride = wandb_config["stride"], 
                                    output_window_size=wandb_config['pred_length'], data_type = wandb_config['data'])
    train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle = False)

    val_data = TimeSeriesDatasetUnique(val_data_tensor, dim_T = dim_T, dim_C = dim_C, dim_O = dim_O,
                                    input_window_size = wandb_config["context_length"] + wandb_config["max_lag"], stride = wandb_config["stride"], 
                                    output_window_size=wandb_config['pred_length'],data_type = wandb_config['data'])
    val_dataloader = DataLoader(val_data, batch_size = batch_size, shuffle = False)

    test_data = TimeSeriesDatasetUnique(test_data_tensor, dim_T = dim_T, dim_C = dim_C, dim_O = dim_O, 
                                    input_window_size = wandb_config["context_length"] + wandb_config["max_lag"], stride = wandb_config["stride"], 
                                    output_window_size=wandb_config['pred_length'],data_type = wandb_config['data'])
    test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle = False)

    config = TimeSeriesTransformerConfig(
        d_model=64,  # Size of the embeddings (and the hidden state)
        encoder_layers=2,  # Number of encoder layers
        decoder_layers=2,  # Number of decoder layers
        encoder_attention_heads=2,  # Number of attention heads in the encoder
        decoder_attention_heads=2,  # Number of attention heads in the decoder
        activation_function='gelu',  # Activation function
        prediction_length=wandb_config['pred_length'],  # Predicting one step ahead
        input_size = dim_input, 
        context_length=wandb_config['context_length'],  # How many steps the encoder looks back
        num_static_categorical_features=0,  # No static categorical features
        num_static_real_features=0,  # No static real features
        num_dynamic_real_features=dynamic_feat,  # All features are dynamic
        num_time_features=dim_time,  # No additional time features
        num_parallel_samples = 10,
        output_hidde_states = True,
        lags_sequence=[i+1 for i in range(wandb_config['max_lag'])] if wandb_config['max_lag'] > 0 else [0],
        scaling=wandb_config['scaling']
    )

    model = TimeSeriesTransformer(config, batch_size = wandb_config["batch_size"], 
                                  window = wandb_config["context_length"]+wandb_config['max_lag'], 
                                  stride = wandb_config["stride"], 
                                  lr = wandb_config["lr"],
                                  dim_T = dim_T, dim_C = dim_C, dim_O = dim_O,
                                  slot_attention=slot_attention,
                                  treatment=treatment, slot_iter=wandb_config['slot_iter'],
                                  num_slot= wandb_config['num_slot'],
                                  transfer_learning = False,
                                  model = wandb_config['model'],
                                  output_window_size = wandb_config['pred_length']) 

    wandb_logger = WandbLogger(project = f'{model_name}_{data_name}', name = f"{epoch}_{context_length}_{wandb_config['max_lag']}_{wandb_config['pred_length']}_{wandb_config['num_slot']}{treatment_txt}", config=wandb_config)


    trainer = pl.Trainer(max_epochs = wandb_config['epoch'], logger = wandb_logger, accelerator=accelerator)
    trainer.fit(model = model, train_dataloaders= train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, test_dataloader)
    trainer.save_checkpoint(f"./weights/{model_name}-{data_name}-{epoch}_{context_length}_{wandb_config['max_lag']}_{wandb_config['pred_length']}_{wandb_config['num_slot']}{treatment_txt}.ckpt")



# %%


