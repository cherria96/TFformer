import sys
sys.path.append("/Users/sujinchoi/Desktop/TF_slot")
from timeseries_dataset_combine import TimeSeriesDatasetUnique, TimeSeriesDataset
from models.Transformer import Model, Config, TimeSeriesForecasting
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
    data_name = "real"
    data_type = "ETTh1"
    
    accelerator = "cpu"
    treatment = True  
    # True: predict treatment using model -> include treatment into input
    kmeans = True  # True: use our model, False: simpel Timeseries Transformer from hugginface
    if kmeans: print('*********** KMEANS clustering ****************')
    slotattention = False
    if slotattention: print('*********** slotattention ****************')

    wandblogging = False
    if kmeans:
        model_name = "ours"
    else:
        model_name = "baseline"

    if treatment:
        treatment_txt = "_treat"
    else:
        treatment_txt = ""
    
    config = Config(
        path= "data/ETTh1.csv",
        data_type= data_name,
        data= data_type,  # "synthetic", "AD"
        seq_len= 96,
        label_len= 48,
        pred_len= 48,
        variate= 'm',
        target= None,
        scale= True,
        is_timeencoded= True,
        random_state= 42,
        output_attention = False,
        enc_in = 7,
        d_model = 512,
        embed = 'fixed',
        freq = 'h',
        dropout = 0.05,
        dec_in = 7,
        factor = 5,
        n_heads = 8,
        d_ff = 1028,
        activation = 'gelu',
        e_layers = 3,
        n_components = 10,
        num_clusters = 5,
        d_layers = 3,
        c_out = 7,
        batch_size= 64,
        epoch= 20,
        lr= 0.0005,
        loss= 'mse',
        scheduler= 'exponential',
        inverse_scaling = True,
        kmeans = kmeans,
        num_workers = 0,
        slotattention=slotattention,
        small_batch_size = 20,
        small_stride = 1)

    wandb_config = config.to_dict()

    train_data = TimeSeriesDataset(
                path = config.path,
                split="train",
                split_sequential = False,
                seq_len=config.seq_len,
                label_len=config.label_len,
                pred_len=config.pred_len,
                variate=config.variate,
                target=config.target,
                scale=config.scale,
                is_timeencoded=config.is_timeencoded,
                frequency=config.freq,
                random_state=config.random_state,
                small_batch_size=config.small_batch_size,
                stride=config.small_stride

            )

    val_data = TimeSeriesDataset(
                path = config.path,
                split="val",
                split_sequential = False,
                seq_len=config.seq_len,
                label_len=config.label_len,
                pred_len=config.pred_len,
                variate=config.variate,
                target=config.target,
                scale=config.scale,
                is_timeencoded=config.is_timeencoded,
                frequency=config.freq,
                random_state=config.random_state,
                small_batch_size=config.small_batch_size,
                stride=config.small_stride

            )

    test_data = TimeSeriesDataset(
                path = config.path,
                split="test",
                split_sequential = False,
                seq_len=config.seq_len,
                label_len=config.label_len,
                pred_len=config.pred_len,
                variate=config.variate,
                target=config.target,
                scale=config.scale,
                is_timeencoded=config.is_timeencoded,
                frequency=config.freq,
                random_state=config.random_state,
                small_batch_size=config.small_batch_size,
                stride=config.small_stride

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
    model = TimeSeriesForecasting(config, scaler = train_data)
    logger = None
    if wandblogging:
        wandb_logger = WandbLogger(project = f'{model_name}_{data_name}', name = f"{config.epoch}_{config.seq_len}_{config.label_len}_{config.pred_len}_{treatment_txt}", config=wandb_config)
        logger = wandb_logger
    trainer = L.Trainer(max_epochs = config.epoch, logger = logger, accelerator=accelerator)
    trainer.fit(model = model, train_dataloaders= train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, test_dataloader)
    trainer.save_checkpoint(f"./weights/{model_name}-{data_name}-{config.epoch}_{config.seq_len}_{config.label_len}_{config.pred_len}_{treatment_txt}.ckpt")








