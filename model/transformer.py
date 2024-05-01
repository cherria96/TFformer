#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
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
    def __init__(self, num_series, d_model=128, nhead=8, num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=256):
        super(TimeSeriesTransformer, self).__init__()

        # Embedding layer for input series
        self.embedding = nn.Linear(num_series, d_model)

        # Transformer layers
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

        # self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        # self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)

        # Output layer to predict next time step
        self.fc_out = nn.Linear(d_model, num_series)

    def forward(self, src, tgt):
        # Embed input
        src_embedded = self.embedding(src)
        tgt_embedded = self.embedding(tgt)

        # Encode and decode
        memory = self.encoder(src_embedded)
        print(memory.shape)
        print(tgt_embedded.shape)
        # output = self.decoder(tgt_embedded, memory)

        # Output layer
        return self.fc_out(output)
import numpy as np
num_points = 4000
A = np.random.normal(size=num_points).astype(np.float32)
B, C, D, E, F, G, H, I, M, N, O, P = [np.zeros(num_points, dtype=np.float32) for _ in range(12)]
# Generate the series according to the relationships
for k in range(3, num_points):
    B[k] = 0.7 * A[k-3] + 0.2 * C[k-1] + np.random.normal()
    C[k] = 0.8 * A[k] + 0.5 * C[k] + np.random.normal()
    O[k] = 0.3 * C[k-5] + np.random.normal()
    P[k] = 0.4 * C[k-1] + 0.1 * P[k] + np.random.normal()
    D[k] = 0.3 * B[k-4] + np.random.normal()

for k in range(2, num_points):
    E[k] = 0.5 * D[k-2] + 0.4 * E[k-2] + np.random.normal()
    F[k] = 0.7 * D[k-2] + np.random.normal()
    M[k] = 0.9 * H[k] + np.random.normal()

for k in range(num_points):
    G[k] = 0.8 * D[k] + np.random.normal()
    I[k] = 0.2 * F[k] + 0.8 * G[k-1] + np.random.normal()
    H[k] = 0.3 * E[k] + np.random.normal()
    N[k] = 0.7 * H[k-1] + np.random.normal()


# Sample dataset
data = np.stack((A, B, C, D, E, F, G, H, I, M, N, O, P), axis=-1)

# Standardize the dataset
mean = data.mean(axis=0)
std = data.std(axis=0)
data = (data - mean) / std

# Dataset parameters
iw = 24*14
ow = 24*7
stride = 1

# Create the dataset and dataloader
time_series_dataset = TimeSeriesDataset(data, iw, ow, stride)
time_series_dataloader = DataLoader(time_series_dataset, batch_size=32, shuffle=True)
#%%
# Initialize the model
num_series = data.shape[-1]
model = TimeSeriesTransformer(num_series)

# Training loop and additional steps
# You may add the training loop with loss function and optimizer here.
# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20  # Define the number of epochs

for epoch in range(num_epochs):
    total_loss = 0
    for X, y in time_series_dataloader:
        # Prepare target for decoder
        tgt = y.unsqueeze(1)  # Reshape to be compatible with model output
        print(X.shape)
        print(tgt.shape)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(X, tgt)

        # Compute loss
        loss = criterion(output.squeeze(), y)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(time_series_dataloader)}')
# %%
