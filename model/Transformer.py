#%%

import torch
torch.set_default_dtype(torch.float32)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import math
import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning import LightningModule
import pytorch_lightning as pl

# Define a custom dataset class

class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_window_size, output_window_size, stride=1):
        self.data = data
        self.feature_dim = self.data.shape[-1]
        self.stride = stride
        self.num_samples = (len(data) - input_window_size- output_window_size) // stride + 1

        X = np.zeros([input_window_size, self.num_samples, self.feature_dim])
        Y = np.zeros([output_window_size, self.num_samples, self.feature_dim])

        for i in np.arange(self.num_samples):
            start_x = self.stride * i
            end_x = start_x + input_window_size
            X[:,i] = data[start_x : end_x]

            start_y = self.stride * i + input_window_size
            end_y = start_y + output_window_size
            Y[:,i] = data[start_y : end_y]
        X = X.reshape(input_window_size, self.num_samples, -1).transpose(1,0,2)
        Y = Y.reshape(output_window_size, self.num_samples, -1).transpose(1,0,2)

        self.x = X
        self.y = Y

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        return torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

# Transformer model for time series prediction
class TimeSeriesTransformer(LightningModule):
    def __init__(self,num_features, iw, ow, d_model, nhead, nlayers, dropout=0.5):
        super(TimeSeriesTransformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers) 
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.encoder = nn.Sequential(
            nn.Linear(num_features, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, d_model)
        )
        
        self.linear =  nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, num_features)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(iw, (iw+ow)//2),
            nn.ReLU(),
            nn.Linear((iw+ow)//2, ow)
        ) 
        self.loss_fn = nn.MSELoss()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_mask):
        # self.src_mask = self.generate_square_subsequent_mask(src.shape[1])
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src.transpose(0,1), src_mask).transpose(0,1)
        output = self.linear(output)
        output = self.linear2(output.permute(0,2,1)).permute(0,2,1)
        return output
    
    def configure_optimizers(self):
        lr = 0.001
        optimizer = optim.Adam(self.parameters(), lr = lr)
        # lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma = 0.99)

        # return [optimizer], [lr_scheduler]
        return optimizer
    
    def training_step(self, batch, trainer=True):
        src = batch[0]
        src_mask = self.generate_square_subsequent_mask(src.shape[1])

        pred = self(src, src_mask)
        actual = batch[1]
        mse_loss = self.loss_fn(pred, actual)
        # mse_loss = (batch['active_entries'] * mse_loss).sum() / batch['active_entries'].sum()
        if trainer:
            self.log(f'train_mse_loss', mse_loss, on_epoch = True, on_step = False, sync_dist = True)
        return mse_loss

    def validation_step(self, batch, batch_idx):
        src = batch[0]
        src_mask = self.generate_square_subsequent_mask(src.shape[1])

        pred = self(src, src_mask)
        actual = batch[1]

        mse_loss = self.loss_fn(pred, actual)
        self.log(f'validation_mse_loss', mse_loss.mean(), on_epoch=True, on_step=False, sync_dist=True)
    

    def test_step(self, batch, batch_idx):
        src = batch[0]
        src_mask = self.generate_square_subsequent_mask(src.shape[1])

        pred = self(src, src_mask)
        actual = batch[1]
        mse_loss = self.loss_fn(pred, actual)
        self.log(f'test_mse_loss', mse_loss.mean(), on_epoch=True, on_step=False, sync_dist=True)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def gen_attention_mask(x):
    mask = torch.eq(x, 0)
    return mask

if __name__ == "__main__":
    data = np.load('/home/user126/TFformer/synthetic_data/data/sbk_AD.npy')
    # Dataset parameters
    iw = 10*7
    ow = 3*7
    stride = 1

    # Create the dataset and dataloader
    train_data = TimeSeriesDataset(data[:int(len(data) * 0.5)], iw, ow, stride)
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_data = TimeSeriesDataset(data[int(len(data) * 0.5):int(len(data) * 0.7)], iw, ow, stride)
    val_dataloader = DataLoader(val_data, batch_size=64, shuffle=True)
    test_data = TimeSeriesDataset(data[int(len(data) * 0.7):], iw, ow, stride)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    # Initialize the model
    num_features = data.shape[-1]
    model = TimeSeriesTransformer(num_features, iw = iw, ow = ow, d_model = 10, nhead = 2, nlayers = 2, dropout = 0.2)
    trainer = pl.Trainer(accelerator = "cpu",max_epochs = 100, log_every_n_steps = 40, logger = None)
    
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader)

    """

    # Training loop and additional steps
    # You may add the training loop with loss function and optimizer here.
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 20  # Define the number of epochs

    for epoch in range(num_epochs):
        total_loss = 0
        for X, y in train_dataloader:
            # Prepare target for decoder
            optimizer.zero_grad()
            src_mask = model.generate_square_subsequent_mask(X.shape[1])
            # Forward pass
            output = model(X, src_mask) # (32, 7)

            # Compute loss
            loss = criterion(output, y)

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            # Accumulate loss
            total_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_dataloader)}')

    model.eval()
    for X, y in val_dataloader:
        src_mask = model.generate_square_subsequent_mask(X.shape[1])
        output = model(X, src_mask)
        loss = criterion(output, y)
        print(loss)
        break
    """
# %%
