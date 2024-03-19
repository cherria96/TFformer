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

class DataCreate(Dataset):
    def __init__(self, A, X, Y, V,output, sequence_lengths):
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
        self.output = output # outcome을 Y에서 가져와야 할 것 같은데 일단 돌아가게만 해볼게

    def __len__(self):
        return len(self.A)

    def __getitem__(self, idx):
        seq_len = self.sequence_lengths[idx]
        active_entries = torch.ones(seq_len)
        
        return {
            'prev_A': self.A[idx, :-1],  # Assuming last treatment is `curr_A`
            'X': self.X[idx],
            'prev_Y': self.Y[idx, :-1],  # Assuming similar structure to A
            'static_inputs': self.V[idx],
            'curr_A': self.A[idx, :-1],
            'active_entries': active_entries,
            'outputs':self.output[idx,:-1]
        }


# Example dimensions
num_samples = 200  # Number of samples in the dataset
seq_length = 60  # Length of each sequence
dim_A = 5  # Dimension of treatments
dim_X = 10  # Dimension of vitals
dim_Y = 1  # Dimension of outputs
dim_V = 3  # Dimension of static inputs
# Simulate data
A = torch.randn(num_samples, seq_length+1, dim_A) # 왜 얘네 (25,59,5)이지? 그래서 +1 해두긴했어
X = torch.randn(num_samples, seq_length, dim_X)
# Y = torch.randint(0, 2, (num_samples, seq_length, dim_Y)).float()  # Binary outcomes
Y = torch.randn(num_samples, seq_length+1, dim_Y)
V = torch.randn(num_samples, dim_V)
output = torch.randn(num_samples,seq_length+1,10)
sequence_lengths = torch.full((num_samples,), seq_length, dtype=torch.long)  # Here, all sequences are of the same length for simplicity

# Initialize the dataset
batch_size = 32
train_dataset = DataCreate(A, X, Y, V,output, sequence_lengths)
test_dataset = DataCreate(A, X, Y, V,output, sequence_lengths)

# Create the DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Example of iterating over the DataLoader
for batch in train_loader:
    # Each 'batch' is a dictionary matching the expected input structure of your model
    print(batch['prev_A'].shape)  # Example: access the 'prev_A' component of the batch
    break  # Break after one iteration for demonstration

trainer = pl.Trainer(accelerator = "cpu",max_epochs = 10)
model = CT(dim_A=dim_A, dim_X = dim_X, dim_Y = dim_Y, dim_V = dim_V)
trainer.fit(model, train_loader)
trainer.test(model,test_loader)