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
train_loader = DataLoader(datasetcollection.train_f, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(datasetcollection.val_f, batch_size=batch_size, shuffle=False)
# # Simulate data
# A = torch.randn(num_samples, seq_length, dim_A) # 왜 얘네 (25,59,5)이지? 그래서 +1 해두긴했어
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
for batch in train_loader:
    # Each 'batch' is a dictionary matching the expected input structure of your model
    print(batch.keys())  # Example: access the 'prev_A' component of the batch
    break  # Break after one iteration for demonstration

trainer = pl.Trainer(accelerator = "cpu",max_epochs = 10)
model = CT(dim_A=dim_A, dim_X = dim_X, dim_Y = dim_Y, dim_V = dim_V)
trainer.fit(model, train_loader)
trainer.test(model,val_loader)
# %%
