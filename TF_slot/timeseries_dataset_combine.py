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
            frequency = 'd'
            ):
        assert split in ['train', 'val', 'test']
        self.path = path 
        self.split = split
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.variate = variate
        self.target = target
        self.scale = scale
        self.random_state = random_state
        self.is_timeencoded = is_timeencoded
        self.frequency = frequency

        self.scaler = StandardScaler()

        self.prepare_data()

    def prepare_data(self):
        df = pd.read_csv(self.path) # 1st col: date 
        df = df.fillna(method = 'ffill')
        df = df.fillna(method = 'bfill')
        df['date'] = pd.to_datetime(df['date'])

        indices = df.index.tolist()
        train_size = 0.5
        val_size = 0.3
        test_size = 0.2
        
        train_indices, temp_indices = train_test_split(indices, train_size=train_size, random_state=self.random_state)
        val_indices, test_indices = train_test_split(temp_indices, test_size=test_size / (val_size + test_size), random_state=self.random_state)
        if self.split == 'train':
            split_indices = train_indices
        elif self.split == 'val':
            split_indices = val_indices
        elif self.split == 'test':
            split_indices = test_indices
        
        split_indices.sort()
        
        df_split = df.loc[split_indices]
        if self.variate == 'm' or self.variate == 'mu':
            data_columns = df.columns[1:]
            df_data = df_split[data_columns]
            self.feature_names = data_columns
        elif self.variate == 'u':
            df_data = df.split[[self.target]]
            self.feature_name = [self.target]
        
        data = torch.FloatTensor(df_data.values)

        if self.scale:
            train_data = df.loc[train_indices][self.feature_names].values
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)
        else:
            data = df_data.values
        
        timestamp = df_split[['date']]
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
            #TBU
            pass

        self.time_series = torch.FloatTensor(data)
        self.timestamp = torch.FloatTensor(timestamp_data)
    
    def __getitem__(self, index):
        x_begin_index = index
        x_end_index = x_begin_index + self.seq_len
        y_begin_index = x_end_index - self.label_len
        y_end_index = y_begin_index + self.label_len + self.pred_len

        x = self.time_series[x_begin_index:x_end_index]
        y = self.time_series[y_begin_index:y_end_index]
        x_timestamp = self.timestamp[x_begin_index:x_end_index]
        y_timestamp = self.timestamp[y_begin_index:y_end_index]

        return x, y, x_timestamp, y_timestamp

    def __len__(self):
        return len(self.time_series) - self.seq_len - self.pred_len + 1

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

                

