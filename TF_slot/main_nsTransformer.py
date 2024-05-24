import sys
sys.path.append("/Users/sujinchoi/Desktop/TF_slot")
from timeseries_dataset_combine import TimeSeriesDatasetUnique, TimeSeriesDataset
from models.Transformer import Model, Config, TimeSeriesForecasting
from models.ns_Transformer import ns_TimeSeriesForecasting
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
import lightning as L

import csv
from sklearn.preprocessing import MinMaxScaler
import torch
torch.set_default_dtype(torch.float32)
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
    data_name = "AD"
    data_type = "sbk_AD"
    size = {
        'seq_len': 30,
        'label_len': 15,
        'pred_len': 15
    }
    d_model = 64
    small_batch_size = 32
    split_sequential = False # True: chronological split; False: random split
    accelerator = "cpu"
    kmeans = False  # True: Our model, False: Baseline
    slotattention = False # True: Our model, False: Baseline
    wandblogging = True
    #####################################
    if kmeans: print('*********** KMEANS clustering ****************')
    if slotattention: print('*********** slotattention ****************')

    if kmeans or slotattention:
        model_name = "ours"
    else:
        model_name = "baseline"

    # if treatment:
    #     treatment_txt = "_treat"
    # else:
    treatment_txt = ""

    '''
    if slot attention applied, 
    c_out = enc_in + 1 where 1 is mask for each slot
    '''
    if data_type == "sbk_AD":
        path = "data/sbk_AD_unscaled.csv"
        enc_in = dec_in = 8
        c_out = enc_in + 1 if slotattention else enc_in
        freq = 'd'
    elif data_type == "ETTm2":
        path = "data/ETTm2.csv"
        enc_in = dec_in = 7
        c_out = enc_in + 1 if slotattention else enc_in
        freq = 't'
    elif data_type == "ETTh1":
        path = "data/ETTh1.csv"
        enc_in = dec_in = 7
        c_out = enc_in + 1 if slotattention else enc_in
        freq = 'h'
    else:
        raise NotImplementedError
    
    

    config = Config(
        path= path,
        data_type= data_name,
        data= data_type,  # "synthetic", "AD"
        seq_len= size['seq_len'],
        label_len= size['label_len'],
        pred_len= size['pred_len'],
        variate= 'm',
        target= None,
        scale= True,
        is_timeencoded= True,
        random_state= 42,
        output_attention = False,
        enc_in = enc_in,
        d_model = d_model,
        embed = 'fixed',
        freq = freq,
        dropout = 0.05,
        dec_in = dec_in,
        factor = 5,
        n_heads = 8,
        d_ff = 128,
        activation = 'gelu',
        e_layers = 3,
        n_components = 10,
        num_clusters = 5,
        d_layers = 3,
        c_out = c_out,
        batch_size= 32,
        epoch= 20,
        lr= 0.0005,
        loss= 'mse',
        scheduler= 'exponential',
        inverse_scaling = True,
        kmeans = kmeans,
        num_workers = 0,
        slotattention=slotattention,
        small_batch_size = small_batch_size,
        small_stride = 1)

    wandb_config = config.to_dict()

    train_data = TimeSeriesDataset(
                path = config.path,
                split="train",
                split_sequential = split_sequential,
                seq_len=config.seq_len,
                label_len=config.label_len,
                pred_len=config.pred_len,
                variate=config.variate,
                target=config.target,
                scale=config.scale,
                is_timeencoded=config.is_timeencoded,
                frequency=config.freq,
                random_state=config.random_state,
                # small_batch_size=config.small_batch_size,
                # stride=config.small_stride

            )

    val_data = TimeSeriesDataset(
                path = config.path,
                split="val",
                split_sequential = split_sequential,
                seq_len=config.seq_len,
                label_len=config.label_len,
                pred_len=config.pred_len,
                variate=config.variate,
                target=config.target,
                scale=config.scale,
                is_timeencoded=config.is_timeencoded,
                frequency=config.freq,
                random_state=config.random_state,
                # small_batch_size=config.small_batch_size,
                # stride=config.small_stride

            )

    test_data = TimeSeriesDataset(
                path = config.path,
                split="test",
                split_sequential = split_sequential,
                seq_len=config.seq_len,
                label_len=config.label_len,
                pred_len=config.pred_len,
                variate=config.variate,
                target=config.target,
                scale=config.scale,
                is_timeencoded=config.is_timeencoded,
                frequency=config.freq,
                random_state=config.random_state,
                # small_batch_size=config.small_batch_size,
                # stride=config.small_stride

            )
    
    train_dataloader = DataLoader(
                        train_data,
                        batch_size=config.batch_size,
                        shuffle=True,
                        num_workers=config.num_workers,
                        )
    val_dataloader = DataLoader(
                        val_data,
                        batch_size=config.batch_size,
                        shuffle=False,
                        num_workers=config.num_workers,
                        )
    test_dataloader = DataLoader(
                        test_data,
                        batch_size=config.batch_size,
                        shuffle=False,
                        num_workers=config.num_workers,
                        )
    model = ns_TimeSeriesForecasting(config, scaler = train_data)
    logger = None
    if wandblogging:
        wandb_logger = WandbLogger(project = f'{model_name}_{data_name}', name = f"ns_Transformer_{config.epoch}_{config.seq_len}_{config.label_len}_{config.pred_len}_{treatment_txt}", config=wandb_config)
        logger = wandb_logger
    trainer = L.Trainer(max_epochs = config.epoch, logger = logger, accelerator=accelerator)
    trainer.fit(model = model, train_dataloaders= train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, test_dataloader)
    trainer.save_checkpoint(f"./weights/{model_name}-{data_name}-{config.epoch}_{config.seq_len}_{config.label_len}_{config.pred_len}_{treatment_txt}.ckpt")








