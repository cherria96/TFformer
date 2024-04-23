#%%
import sys
sys.path.append("/Users/sujinchoi/Desktop/TFformer")
import pytorch_lightning as pl
from model.CT_ourmodel import CT
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from src.data.cancer_sim.dataset import SyntheticCancerDatasetCollection
from realdata.realdatas import WeatherRealDatasetCollection, VARSyntheticDatasetCollection
import logging
import datetime
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from model.utils import unroll_temporal_data
#%%
seed_everything(100)

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

class Trainers:
    def __init__(self, datasetcollection, config, dim_vars):

        torch.set_default_dtype(torch.float64)

        # cancer_sim 
        self.datasetcollection = datasetcollection
        self.datasetcollection.process_data_multi()
        self.config = config
        self.batch_size = self.config['batch_size']
        self.epoch = self.config['epochs']
        self.fc_hidden_units = self.config['fc_hidden_units']

        self.window_len = self.config["window_len"]
        self.t_step = self.config["t_step"]
        self.keys = ['prev_treatments', 'current_treatments', 'outputs', 'active_entries', 'unscaled_outputs', 'prev_outputs']
        if self.config["has_vital"]:
            self.keys.append(['vitals', 'next_vitals'])
        self.train_loader = None
        self.val_loader = None
        wandb.login(key="aa1e46306130e6f8863bbad2d35c96d0a62a4ddd")
        self.wandb_logger = WandbLogger(project = 'TFFormer', name = f'CT_VAR4_unroll_{self.batch_size}_{self.epoch}_{self.window_len}_{self.t_step}')
        self.dim_A = dim_vars["A"]
        self.dim_X = dim_vars["X"]
        self.dim_V = dim_vars["V"]
        self.dim_Y = dim_vars["Y"]

    def _unroll_data(self):
        print("=====Start unroll data processing====")

        for key in self.keys:
            observed_nodes_list= list(range(self.datasetcollection.train_f.data[key].shape[-1]))
            self.datasetcollection.train_f.data[key],_,_ = unroll_temporal_data(self.datasetcollection.train_f.data[key], 
                                                                                observed_nodes_list, window_len = self.window_len,t_step = self.t_step)
            if key == 'prev_treatments':
                self.dim_A = self.datasetcollection.train_f.data[key].shape[-1] # Dimension of treatments
            elif key == 'vitals':
                self.dim_X = self.datasetcollection.train_f.data[key].shape[-1] # Dimension of vitals
            elif key == "outputs":
                self.dim_Y = self.datasetcollection.train_f.data[key].shape[-1] # Dimension of outputs
        for key in self.keys:
            observed_nodes_list= list(range(self.datasetcollection.val_f.data[key].shape[-1]))
            self.datasetcollection.val_f.data[key],_,_ = unroll_temporal_data(self.datasetcollection.val_f.data[key], 
                                                                              observed_nodes_list, window_len = self.window_len,
                                                                              t_step = self.t_step)
    def _train_loader(self):
        if self.config["unroll_data"]:
            self._unroll_data()
        return DataLoader(self.datasetcollection.train_f, batch_size=self.batch_size, shuffle=True)
    
    def _val_loader(self):
        if self.config["unroll_data"]:
            self._unroll_data()
        return DataLoader(self.datasetcollection.val_f, batch_size=self.batch_size, shuffle=False)

    def training(self):
        print("=====Start training====")
        self.train_loader = self._train_loader()
        self.val_loader = self._val_loader()
        trainer = pl.Trainer(accelerator = "cpu",max_epochs = self.epoch, log_every_n_steps = 40, logger = self.wandb_logger)
        self.model = CT(dim_A=self.dim_A, dim_X = self.dim_X, 
                        dim_Y = self.dim_Y, dim_V = self.dim_V, fc_hidden_units=self.fc_hidden_units)
        trainer.fit(self.model, self.train_loader, self.val_loader)
        trainer.test(self.model, self.val_loader)
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")   
        trainer.save_checkpoint(f"weights/unroll_{self.window_len}_{self.t_step}_{now}.pt")
        self.val_rmse_orig, self.val_rmse_all = self.model.get_normalised_masked_rmse(self.datasetcollection.val_f)
        logger.info(f'Val normalised RMSE (all): {self.val_rmse_all}; Val normalised RMSE (orig): {self.val_rmse_orig}')

if __name__=="__main__":
    config = {
    "lr" : 0.01,
    "epochs" : 150,
    "batch_size": 64,
    "fc_hidden_units": 32,
    "has_vital": True,
    "unroll_data": False,
    "window_len": 3,
    "t_step": 3
    }
    dim_vars = {
        "A": 2,
        "X": 5,
        "V": 2,
        "Y": 1
    }
    file_path = "synthetic-data/data/VAR4_2.h5"
    datasetcollection = VARSyntheticDatasetCollection(file_path,dim_treatment = 2, dim_covariates= 5, dim_outcome=1, dim_static=2 )
    trainers = Trainers(datasetcollection, config, dim_vars)
    trainers.training()


