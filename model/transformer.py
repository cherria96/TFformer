#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import math
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
class TimeSeriesTransformer(nn.Module):
    def __init__(self,iw, ow, d_model, nhead, nlayers, dropout=0.5):
        super(TimeSeriesTransformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers) 
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.encoder = nn.Sequential(
            nn.Linear(13, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, d_model)
        )
        
        self.linear =  nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 13)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(iw, (iw+ow)//2),
            nn.ReLU(),
            nn.Linear((iw+ow)//2, ow)
        ) 

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, srcmask):
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src.transpose(0,1), srcmask).transpose(0,1)
        output = self.linear(output)
        output = self.linear2(output.permute(0,2,1)).permute(0,2,1)
        return output

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

import numpy as np
num_points = 4000
A = np.random.normal(size=num_points).astype(np.float32)
B, C, D, E, F, G, H, I, M, N, O, P = [np.zeros(num_points, dtype=np.float32) for _ in range(12)]
# Generate the series according to the relationships
for k in range(3, num_points):
    B[k] = 0.7 * A[k-3] + 0.2 * C[k-1] + 0.05 * np.random.normal()
    C[k] = 0.8 * A[k] + 0.5 * C[k] + 0.05 * np.random.normal()
    O[k] = 0.3 * C[k-5] + 0.05 * np.random.normal()
    P[k] = 0.4 * C[k-1] + 0.1 * P[k] + 0.05 * np.random.normal()
    D[k] = 0.3 * B[k-4] + 0.05 * np.random.normal()

for k in range(2, num_points):
    E[k] = 0.5 * D[k-2] + 0.4 * E[k-2] + 0.05 * np.random.normal()
    F[k] = 0.7 * D[k-2] + 0.05 * np.random.normal()
    M[k] = 0.9 * H[k] + 0.05 * np.random.normal()

for k in range(num_points):
    G[k] = 0.8 * D[k] + 0.05 * np.random.normal()
    I[k] = 0.2 * F[k] + 0.8 * G[k-1] + 0.05 * np.random.normal()
    H[k] = 0.3 * E[k] + 0.05 * np.random.normal()
    N[k] = 0.7 * H[k-1] + 0.05 * np.random.normal()


# Sample dataset
data = np.stack((A, B, C, D, E, F, G, H, I, M, N, O, P), axis=-1)

# Standardize the dataset
mean = data.mean(axis=0)
std = data.std(axis=0)
data = (data - mean) / std

# Dataset parameters
iw = 3*7
ow = 1*7
stride = 1

# Create the dataset and dataloader
train_data = TimeSeriesDataset(data[:int(len(data) * 0.5)], iw, ow, stride)
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
val_data = TimeSeriesDataset(data[int(len(data) * 0.5):int(len(data) * 0.7)], iw, ow, stride)
val_dataloader = DataLoader(val_data, batch_size=32, shuffle=True)
test_data = TimeSeriesDataset(data[int(len(data) * 0.7):], iw, ow, stride)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)
#%%
# Initialize the model
num_series = data.shape[-1]
model = TimeSeriesTransformer(iw = iw, ow = ow, d_model = 10, nhead = 2, nlayers = 2, dropout = 0.2)

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
output = output.detach().numpy()
result = output * std + mean
real = y * std + mean
import matplotlib.pyplot as plt

plt.figure(figsize=(20,5))
plt.plot(range(400,744),real, label="real")
plt.plot(range(744-24*7,744),result, label="predict")
plt.legend()
plt.show()
# %%
