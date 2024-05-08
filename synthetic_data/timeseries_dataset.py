import torch
from torch.utils.data import Dataset
import numpy as np

class _TimeSeriesDataset(Dataset):
    def __init__(self, data, dim_T, dim_C, dim_O, input_window_size, output_window_size, stride=1):
        self.data = data
        self.dim_T = dim_T
        self.dim_C = dim_C
        self.dim_O = dim_O

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
        x = torch.tensor(self.x[idx])
        y = torch.tensor(self.y[idx])
        batch = {
            'prev_treatments': x[:,:self.dim_T],
            'prev_covariates': x[:,self.dim_T:self.dim_C],
            'prev_outcomes': x[:,self.dim_C:],
            'curr_treatments': y[:,:self.dim_T],
            'curr_covariates': y[:,self.dim_T:self.dim_C],
            'curr_outcomes': y[:,self.dim_C:],
        }

        return batch
class TimeSeriesDataset(Dataset):
    def __init__(self, data, dim_T, dim_C, dim_O, window_size, stride=1, data_type = 'AD'):
        self.data = data
        self.dim_T = dim_T
        self.dim_C = dim_C
        self.dim_O = dim_O
        self.total = self.dim_T + self.dim_C - 1 + self.dim_O 
        self.feature_dim = self.data.shape[-1]
        self.stride = stride
        self.data_type = data_type
        self.num_samples = (len(data) - window_size) // stride + 1

        _data = np.zeros([window_size, self.num_samples, self.feature_dim])

        for i in np.arange(self.num_samples):
            start_x = self.stride * i
            end_x = start_x + window_size
            _data[:,i] = self.data[start_x : end_x]
        X  = _data[:-1].transpose(1, 0, 2)
        Y  = _data[1:].transpose(1,0,2)

        self.x = X
        self.y = Y

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx])
        y = torch.tensor(self.y[idx])
        if self.data_type == 'AD':
            batch = {
                'prev_treatments': x[:, :self.dim_T],
                'prev_covariates': x[:, self.dim_T : self.dim_T+self.dim_C-1],
                'prev_timefeatures': x[:, self.dim_T+self.dim_C-1],
                'prev_outcomes': x[:, self.dim_T+self.dim_C : self.total + 1],
                'prev_raw': x[:, self.total + 1 : 2 * self.total + 1],
                'prev_mask': x[:, 2 * self.total + 1 : ],
                'curr_treatments': y[:, : self.dim_T],
                'curr_covariates': y[:, self.dim_T : self.dim_T+self.dim_C-1],
                'curr_timefeatures': x[:, self.dim_T+self.dim_C-1],
                'curr_outcomes': y[:, self.dim_T+self.dim_C : self.total+1],
                'curr_raw': y[:, self.total + 1 : 2 * self.total + 1],
                'curr_mask': y[:, 2 * self.total + 1 : ],
            }
        elif self.data_type == 'synthetic':

            batch= {
                'prev_treatments': x[:, :self.dim_T],
                'prev_covariates': x[:, self.dim_T : self.dim_T+self.dim_C],
                'prev_outcomes': x[:, self.dim_T+self.dim_C : -1],
                'prev_mask': torch.ones_like(x[:,:-1]),
                'prev_timefeatures': x[:,-1],

                'curr_treatments': y[:, : self.dim_T],
                'curr_covariates': y[:, self.dim_T : self.dim_T+self.dim_C],
                'curr_outcomes': y[:, self.dim_T+self.dim_C : -1],
                'curr_mask': torch.ones_like(y[:,:-1]),
                'curr_timefeatures': y[:,-1],
            }

        return batch
