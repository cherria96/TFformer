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
import logging
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
num_patients = {'train': 1000, 'val': 1000, 'test': 100}
datasetcollection = SyntheticCancerDatasetCollection(chemo_coeff = 3.0, radio_coeff = 3.0, num_patients = num_patients, window_size =15, 
                                    max_seq_length = 60, projection_horizon = 5, 
                                    seed = 42, lag = 0, cf_seq_mode = 'sliding_treatment', treatment_mode = 'multiclass')
datasetcollection.process_data_multi()

# Example dimensions
# num_samples = 1000  # Number of samples in the dataset
# seq_length = 60  # Length of each sequence
dim_A = 4  # Dimension of treatments
dim_X = 0  # Dimension of vitals
dim_Y = 1  # Dimension of outputs
dim_V = 1  # Dimension of static inputs
batch_size = 32
epoch = 100
train_loader = DataLoader(datasetcollection.train_f, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(datasetcollection.val_f, batch_size=batch_size, shuffle=False)

# Example of iterating over the DataLoader
for batch in train_loader:
    # Each 'batch' is a dictionary matching the expected input structure of your model
    print(batch.keys())  # Example: access the 'prev_A' component of the batch
    break  # Break after one iteration for demonstration

#%%
checkpoint_path = "C:/Users/user/Documents/causaltf/TFformer/weights/10000_100_1_256.pt"
model = CT.load_from_checkpoint(checkpoint_path)
trainer = pl.Trainer(accelerator="cpu", max_epochs=1)
trainer.test(model, val_loader)

val_rmse_orig, val_rmse_all = model.get_normalised_masked_rmse(datasetcollection.val_f)
logger.info(f'Val normalised RMSE (all): {val_rmse_all}; Val normalised RMSE (orig): {val_rmse_orig}')

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