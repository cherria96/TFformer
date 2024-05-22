import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler

# non-overlapping dataset, different iput, output window size is available
class TimeSeriesDatasetUnique(Dataset):
    def __init__(self, data, dim_T, dim_C, dim_O, input_window_size, output_window_size, stride=1, data_type = 'AD'):
        self.data = data
        self.dim_T = dim_T
        self.dim_C = dim_C
        self.dim_O = dim_O

        self.total = self.dim_T + self.dim_C + self.dim_O 
        self.data_type = data_type

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

        
        if self.data_type == 'AD':
            # need to check
            batch = {
                'prev_treatments': x[:, :self.dim_T],
                'prev_covariates': x[:, self.dim_T : self.dim_T+self.dim_C],
                'prev_outcomes': x[:, self.dim_T+self.dim_C : self.total],
                'prev_mask': x[:, 2*self.total+3: ],
                'prev_timefeatures': x[:, self.total:self.total+3],   # among Day, Month, Weekday, utilize only Day

                'curr_treatments': y[:, : self.dim_T],
                'curr_covariates': y[:, self.dim_T : self.dim_T+self.dim_C],
                'curr_outcomes': y[:, self.dim_T+self.dim_C : self.total],
                'curr_mask': y[:, 2*self.total+3: ],
                'curr_timefeatures': y[:, self.total:self.total+3],
            }
        else:  # synthetic, real goes to here
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
    
class TimeSeriesDatasetOverlap(Dataset):
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

class _TimeSeriesDataset(Dataset):
    def __init__(self, data, dim_T, dim_C, dim_O, split = 'train',
                 size = None, scaling = True, stride = 1, data_type = 'AD'):
        # size [seq_len, label_len, pred_len]
        self.data = data
        self.dim_T = dim_T
        self.dim_C = dim_C
        self.dim_O = dim_O

        self.total = self.dim_T + self.dim_C + self.dim_O 
        self.data_type = data_type

        self.feature_dim = self.data.shape[-1]
        self.stride = stride

        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        self.num_samples = (len(data) - self.seq_len - self.pred_len) + 1

        if scaling:
            self.scaler = StandardScaler



        X = np.zeros([self.seq_len, self.num_samples, self.feature_dim])
        Y = np.zeros([self.label_len + self.pred_len, self.num_samples, self.feature_dim])

        for i in np.arange(self.num_samples):
            start_x = self.stride * i
            end_x = start_x + self.seq_len
            X[:,i] = data[start_x : end_x]

            start_y = end_x - self.label_len
            end_y = start_y + self.label_len + self.pred_len
            Y[:,i] = data[start_y : end_y]
        X = X.reshape(self.seq_len, self.num_samples, -1).transpose(1,0,2)
        Y = Y.reshape(self.label_len + self.pred_len, self.num_samples, -1).transpose(1,0,2)

        self.x = X
        self.y = Y

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx])
        y = torch.tensor(self.y[idx])

        
        if self.data_type == 'AD':
            # need to check
            batch = {
                'prev_treatments': x[:, :self.dim_T],
                'prev_covariates': x[:, self.dim_T : self.dim_T+self.dim_C],
                'prev_outcomes': x[:, self.dim_T+self.dim_C : self.total],
                'prev_mask': x[:, 2*self.total+3: ],
                'prev_timefeatures': x[:, self.total:self.total+3],   # among Day, Month, Weekday, utilize only Day

                'curr_treatments': y[:, : self.dim_T],
                'curr_covariates': y[:, self.dim_T : self.dim_T+self.dim_C],
                'curr_outcomes': y[:, self.dim_T+self.dim_C : self.total],
                'curr_mask': y[:, 2*self.total+3: ],
                'curr_timefeatures': y[:, self.total:self.total+3],
            }
        else:  # synthetic, real goes to here
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

import pandas as pd
from sklearn.model_selection import train_test_split
from utils.timefeatures import time_features

class TimeSeriesDataset(Dataset):
    def __init__(
            self,
            path,
            split = 'train',
            split_sequential = False, # True: chronological split; False: random split -> TBU
            seq_len = 24 * 4 * 4,
            label_len = 24 * 4,
            pred_len = 24 * 4, 
            variate = 'u',
            target = 'OT',
            scale = True,
            random_state = 42,
            is_timeencoded = False,
            frequency = 'd',
            small_batch_size = None,
            stride = 1,
            ):
        assert split in ['train', 'val', 'test']
        self.path = path 
        self.split = split
        self.split_sequential = split_sequential
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.variate = variate
        self.target = target
        self.scale = scale
        self.random_state = random_state
        self.is_timeencoded = is_timeencoded
        self.frequency = frequency
        self.small_batch_size = small_batch_size
        self.stride = stride
        
        self.scaler = StandardScaler()

        self.prepare_data()

    def prepare_data(self):
        df = pd.read_csv(self.path) # 1st col: date 
        df = df.ffill()
        df = df.bfill()
        df['date'] = pd.to_datetime(df['date'])
        # indices = df.index.tolist()
        train_size = 0.5
        val_size = 0.3
        test_size = 0.2

        self.feature_dim = df.shape[-1]
        self.num_samples = (len(df) - self.seq_len - self.label_len) // self.stride + 1

        timestamp = df[['date']]
        if not self.is_timeencoded:
            timestamp['month'] = timestamp.date.apply(lambda row: row.month, 1)
            timestamp['day'] = timestamp.date.apply(lambda row: row.day, 1)
            timestamp['weekday'] = timestamp.date.apply(lambda row: row.weekday(), 1)
            if self.frequency == 'h' or self.frequency == 'm':
                timestamp['hour'] = timestamp.date.apply(lambda row: row.hour, 1)
            if self.frequency == 'm':
                timestamp['minute'] = timestamp.date.apply(lambda row: row.minute, 1)
                timestamp['minute'] = timestamp.minute.map(lambda x: x // 15)
            timestamp_data = timestamp.drop('date', axis = 1).values
        else:
            timestamp_data = time_features(pd.to_datetime(timestamp.date.values), freq = self.frequency)
            timestamp_data = timestamp_data.transpose(1,0)

        data_all = np.zeros([self.seq_len + self.pred_len, self.num_samples, self.feature_dim - 1])
        timestamp_list = []
        for i in np.arange(self.num_samples):
            start_index = self.stride * i
            end_index = start_index + self.seq_len + self.pred_len
            data_all[:,i] = df.iloc[start_index : end_index, 1:]
            timestamp_list.append(timestamp_data[start_index:end_index])
        data_all = data_all.reshape(-1, self.num_samples, self.feature_dim - 1).transpose(1,0,2) # (17277, 144, 7)
        timestamp = np.stack(timestamp_list, axis = 0) # (17277,144)

        indices = np.arange(len(data_all))
        # (n,w,f) -> (n', b, w, f)
        if self.split_sequential:
            train_end = int(len(indices) * train_size)
            val_end = train_end + int(len(indices) * val_size)
            train_indices = indices[:train_end]
            if self.split == 'train':
              split_indices = train_indices
            elif self.split == 'val':
              split_indices = indices[train_end:val_end]
            if self.split == 'test':
              split_indices = indices[val_end:]
        else:
            train_indices, temp_indices = train_test_split(indices, train_size=train_size, random_state=self.random_state)
            val_indices, test_indices = train_test_split(temp_indices, test_size=test_size / (val_size + test_size), random_state=self.random_state)
            if self.split == 'train':
                split_indices = train_indices
            elif self.split == 'val':
                split_indices = val_indices
            elif self.split == 'test':
                split_indices = test_indices
            split_indices.sort() 
        df_split = data_all[split_indices]
        timestamp_data = timestamp[split_indices]
        if self.variate == 'm' or self.variate == 'mu':
            df_data = df_split
            # self.feature_names = data_columns
            self.feature_list = np.arange(df_data.shape[-1]).tolist()
        elif self.variate == 'u':
            target_idx = df.columns.get_loc(self.target)
            # df_data = df_split[[self.target]]
            df_data = df_split[:,:, target_idx]
            self.feature_list = [target_idx]
        
        data = torch.FloatTensor(df_data)
        data_shape = data.shape

        if self.scale:
            train_data = data_all[train_indices][:,:,self.feature_list]
            # train_data = df.loc[train_indices][self.feature_names].values
            self.scaler.fit(train_data.reshape(-1, data_shape[2]))
            data = self.scaler.transform(data.reshape(-1, data_shape[2]))
            data = data.reshape(data_shape[0], data_shape[1], data_shape[2])

        self.time_series = torch.FloatTensor(data)
        self.timestamp = torch.FloatTensor(timestamp_data)
        
    
    def __getitem__(self, index):
        begin_index = index
        end_index = begin_index + self.small_batch_size
        small_batch = self.time_series[begin_index : end_index]
        small_batch_timestamp = self.timestamp[begin_index : end_index]

        x = small_batch[:,:self.seq_len]
        y = small_batch[:, self.seq_len - self.label_len: self.seq_len + self.pred_len]
        x_timestamp = small_batch_timestamp[:,:self.seq_len]
        y_timestamp = small_batch_timestamp[:, self.seq_len - self.label_len: self.seq_len + self.pred_len]

        return x, y, x_timestamp, y_timestamp

    def __len__(self):
        # return len(self.time_series) - self.seq_len - self.pred_len + 1
        return len(self.time_series) - self.small_batch_size + 1

    def inverse_transform(self, data):
        if data.ndim >=3:
            data_shape = data.shape
            data = data.reshape(-1, data.shape[-1])
            data = self.scaler.inverse_transform(data.cpu().detach().numpy())
            data = torch.tensor(data).reshape(*data_shape)

        return data

    @property
    def num_features(self):
        return self.time_series.shape[1]

    @property
    def columns(self):
        return self.feature_names

                

