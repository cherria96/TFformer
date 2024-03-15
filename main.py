import pytorch_lightning as pl
from model.CT_ourmodel import CT
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader

class DataCreate(Dataset):
    def __init__(self, A, X, Y, V, sequence_lengths):
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

    def __len__(self):
        return len(self.treatments)

    def __getitem__(self, idx):
        seq_len = self.sequence_lengths[idx]
        active_entries = torch.zeros(self.max_seq_length)  # Assuming max_seq_length is defined
        active_entries[:seq_len] = 1
        
        return {
            'prev_A': self.A[idx, :-1],  # Assuming last treatment is `curr_A`
            'X': self.X[idx],
            'prev_Y': self.Y[idx, :-1],  # Assuming similar structure to A
            'static_inputs': self.V[idx],
            'curr_A': self.A[idx, -1],
            'active_entries': active_entries
        }

A = None
X = None
Y = None
V = None
sequence_lengths= None
neural_mass = DataCreate(A,X,Y,V,sequence_lengths)
train_loader = DataLoader(neural_mass, batch_size=32, shuffle=True)
trainer = pl.Trainer()
model = CT()
trainer.fit(model, train_loader)