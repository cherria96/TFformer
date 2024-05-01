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
from realdata import WeatherRealDatasetCollection, VARSyntheticDatasetCollection
import logging
import datetime
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything

#%%
seed_everything(100)

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

'''
class DataCreate(Dataset):
    def __init__(self, A, X, Y, V,sequence_lengths):
        """
        Initialize dataset.
        
        Parameters:
        - A: Tensor of A.
        - X: Tensor of X.
        - outputs: Tensor of outputs.
        - static_inputs: Tensor of static features.
        - sequence_lengths: List or tensor of sequence lengths for each sample.
        """
        self.A = A
        self.X = X
        self.Y = Y
        self.V = V
        self.sequence_lengths = sequence_lengths
        # self.output = output # outcome을 Y에서 가져와야 할 것 같은데 일단 돌아가게만 해볼게

    def __len__(self):
        return len(self.A)

    def __getitem__(self, idx):
        seq_len = self.sequence_lengths[idx]
        outputs = self.Y[:,1:, np.newaxis] 
        active_entries = np.zeros(outputs.shape)

        for i in range(sequence_lengths.shape[0]):
            sequence_length = int(sequence_lengths[i])
            active_entries[i, :sequence_length, :] = 1
        active_entries = torch.ones(seq_len)
        

        return {
            'prev_A': self.A[idx, :-1],  # (num_samples, max_seq_length - 1, dim_A)
            'X': self.X[idx, :-1],
            'prev_Y': self.Y[idx, :-1],  # (num_samples, max_seq_length -1, dim_Y)
            'static_inputs': self.V[idx], # (num_samples, dim_V)
            'curr_A': self.A[idx, 1:].unsqueeze(0), # (num_samples, max_seq_length - 1, dim_A)
            'active_entries': active_entries, # (num_samples, max_seq_length - 1, 1)
            'outputs':self.Y[idx,1:, np.newaxis] # (num_samples, max_seq_length -1, dim_Y)
        }
'''
torch.set_default_dtype(torch.float64)

# cancer_sim 
num_patients = {'train': 10000, 'val': 1000, 'test': 100}
# datasetcollection = SyntheticCancerDatasetCollection(chemo_coeff = 3.0, radio_coeff = 3.0, num_patients = num_patients, window_size =15, 
#                                     max_seq_length = 60, projection_horizon = 5, 
#                                     seed = 42, lag = 0, cf_seq_mode = 'sliding_treatment', treatment_mode = 'multiclass')
# datasetcollection = WeatherRealDatasetCollection(path = 'realdata/weather/weather_.csv',max_seq_length=None)
datasetcollection = VARSyntheticDatasetCollection(path = 'synthetic-data/data/VAR4_dataset.npy',max_seq_length=None)
datasetcollection.process_data_multi()
#%%
# def collate_fn_float32(batch):
#     # Convert all tensors in the batch to float32
#     batch_float32 = [{k: v.to(torch.float32) if torch.is_tensor(v) else v for k, v in item.items()} for item in batch]
#     return torch.utils.data.dataloader.default_collate(batch_float32)

# Example dimensions
# num_samples = 1000  # Number of samples in the dataset
# seq_length = 60  # Length of each sequence
# # Simulate data
# A = torch.randn(num_samples, seq_length, dim_A) 
# X = torch.randn(num_samples, seq_length, dim_X)
# # Y = torch.randint(0, 2, (num_samples, seq_length, dim_Y)).float()  # Binary outcomes
# Y = torch.randn(num_samples, seq_length, dim_Y)
# V = torch.randn(num_samples, dim_V)
# # output = torch.randn(num_samples,seq_length+1,10)
# sequence_lengths = torch.full((num_samples,), seq_length, dtype=torch.long)  # Here, all sequences are of the same length for simplicity

# # Initialize the dataset
# train_dataset = DataCreate(A, X, Y, V, sequence_lengths)
# test_dataset = DataCreate(A, X, Y, V,sequence_lengths)

# # Create the DataLoader
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Example of iterating over the DataLoader
config = {
    "lr" : 0.0001,
    "epochs" : 400,
    "batch_size": 64,
    "fc_hidden_units": 64
}
# dim_A = 4  # Dimension of treatments
# dim_X = 2  # Dimension of vitals
# dim_Y = 3  # Dimension of outputs
# dim_V = 1  # Dimension of static inputs
dim_A = 2  # Dimension of treatments
dim_X = 2  # Dimension of vitals
dim_Y = 1  # Dimension of outputs
dim_V = 1  # Dimension of static inputs
batch_size = config['batch_size']
epoch = config['epochs']
fc_hidden_units = config['fc_hidden_units']
train_loader = DataLoader(datasetcollection.train_f, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(datasetcollection.val_f, batch_size=batch_size, shuffle=False)

wandb.login(key="aa1e46306130e6f8863bbad2d35c96d0a62a4ddd")
wandb_logger = WandbLogger(project = 'TFFormer', name = f'Syntheis_VAR4_wostatic_{fc_hidden_units}_{batch_size}_{epoch}')
# run = wandb.init(
#     name = "CT_256_10", ## Wandb creates random run names if you skip this field
#     reinit = True, ### Allows reinitalizing runs when you re-run this cell
#     # run_id = ### Insert specific run id here if you want to resume a previous run
#     # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
#     project = "TFFormer", ### Project should be created in your wandb account
#     config = config ### Wandb Config for your run
# )

trainer = pl.Trainer(accelerator = "cpu",max_epochs = epoch, log_every_n_steps = 40, logger = wandb_logger)
model = CT(dim_A=dim_A, dim_X = dim_X, dim_Y = dim_Y, dim_V = dim_V, fc_hidden_units=fc_hidden_units)
trainer.fit(model, train_loader, val_loader)

trainer.test(model,val_loader)

#%%
# checkpoint_path = None
# model = CT.load_from_checkpoint(checkpoint_path)
# val_rmse_orig, val_rmse_all = model.get_normalised_masked_rmse(datasetcollection.val_f)
# logger.info(f'Val normalised RMSE (all): {val_rmse_all}; Val normalised RMSE (orig): {val_rmse_orig}')

# encoder_results = {}
# if hasattr(datasetcollection, 'test_cf_one_step'):  # Test one_step_counterfactual rmse
#     test_rmse_orig, test_rmse_all, test_rmse_last = model.get_normalised_masked_rmse(datasetcollection.test_cf_one_step,
#                                                                                             one_step_counterfactual=True)
#     logger.info(f'Test normalised RMSE (all): {test_rmse_all}; '
#                 f'Test normalised RMSE (orig): {test_rmse_orig}; '
#                 f'Test normalised RMSE (only counterfactual): {test_rmse_last}')
#     encoder_results = {
#         'encoder_val_rmse_all': val_rmse_all,
#         'encoder_val_rmse_orig': val_rmse_orig,
#         'encoder_test_rmse_all': test_rmse_all,
#         'encoder_test_rmse_orig': test_rmse_orig,
#         'encoder_test_rmse_last': test_rmse_last
#     }
# elif hasattr(datasetcollection, 'test_f'):  # Test factual rmse
#     test_rmse_orig, test_rmse_all = model.get_normalised_masked_rmse(datasetcollection.test_f)
#     logger.info(f'Test normalised RMSE (all): {test_rmse_all}; '
#                 f'Test normalised RMSE (orig): {test_rmse_orig}.')
#     encoder_results = {
#         'encoder_val_rmse_all': val_rmse_all,
#         'encoder_val_rmse_orig': val_rmse_orig,
#         'encoder_test_rmse_all': test_rmse_all,
#         'encoder_test_rmse_orig': test_rmse_orig
#     }
# %%

now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
trainer.save_checkpoint(f"weights/{num_patients['train']}_{num_patients['test']}_{epoch}_{batch_size}_{now}.pt")

raise NotImplemented
val_rmse_orig, val_rmse_all = model.get_normalised_masked_rmse(datasetcollection.val_f)
logger.info(f'Val normalised RMSE (all): {val_rmse_all}; Val normalised RMSE (orig): {val_rmse_orig}')
results = {}
encoder_results = {}
if hasattr(datasetcollection, 'test_cf_one_step'):  # Test one_step_counterfactual rmse
    test_rmse_orig, test_rmse_all, test_rmse_last = model.get_normalised_masked_rmse(datasetcollection.test_cf_one_step,
                                                                                            one_step_counterfactual=True)
    logger.info(f'Test normalised RMSE (all): {test_rmse_all}; '
                f'Test normalised RMSE (orig): {test_rmse_orig}; '
                f'Test normalised RMSE (only counterfactual): {test_rmse_last}')
    encoder_results = {
        'encoder_val_rmse_all': val_rmse_all,
        'encoder_val_rmse_orig': val_rmse_orig,
        'encoder_test_rmse_all': test_rmse_all,
        'encoder_test_rmse_orig': test_rmse_orig,
        'encoder_test_rmse_last': test_rmse_last
    }
    print(encoder_results)
elif hasattr(datasetcollection, 'test_f'):  # Test factual rmse
    test_rmse_orig, test_rmse_all = model.get_normalised_masked_rmse(datasetcollection.test_f)
    logger.info(f'Test normalised RMSE (all): {test_rmse_all}; '
                f'Test normalised RMSE (orig): {test_rmse_orig}.')
    encoder_results = {
        'encoder_val_rmse_all': val_rmse_all,
        'encoder_val_rmse_orig': val_rmse_orig,
        'encoder_test_rmse_all': test_rmse_all,
        'encoder_test_rmse_orig': test_rmse_orig
    }
    print(encoder_results)
results.update(encoder_results)
test_rmses = {}
if hasattr(datasetcollection, 'test_cf_treatment_seq'):  # Test n_step_counterfactual rmse
    test_rmses = model.get_normalised_n_step_rmses(datasetcollection.test_cf_treatment_seq)
elif hasattr(datasetcollection, 'test_f_multi'):  # Test n_step_factual rmse
    test_rmses = model.get_normalised_n_step_rmses(datasetcollection.test_f_multi)
test_rmses = {f'{k+2}-step': v for (k, v) in enumerate(test_rmses)}

logger.info(f'Test normalised RMSE (n-step prediction): {test_rmses}')
decoder_results = {
    'decoder_val_rmse_all': val_rmse_all,
    'decoder_val_rmse_orig': val_rmse_orig
}
decoder_results.update({('decoder_test_rmse_' + k): v for (k, v) in test_rmses.items()})
