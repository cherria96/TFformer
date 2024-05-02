from torch import nn
import torch

class BROutcomeHead:
    '''
    utils.py: BRTreatmentOutcomeHead.build_outcome
    '''
    def __init__(self, seq_hidden_units, br_size, fc_hidden_units, dim_treatments, dim_outcome, alpha=0.0, update_alpha=True,
                balancing='grad_reverse'):
        super().__init__()
        self.seq_hidden_units = seq_hidden_units
        self.br_size = br_size
        self.fc_hidden_units = fc_hidden_units
        self.dim_treatments = dim_treatments
        self.dim_outcome = dim_outcome
        self.alpha = alpha if not update_alpha else 0.0
        self.alpha_max = alpha
        self.balancing = balancing

        self.linear1 = nn.Linear(self.seq_hidden_units, self.br_size)
        self.elu1 = nn.ELU()

        self.linear2 = nn.Linear(self.br_size, self.fc_hidden_units)
        self.elu2 = nn.ELU()
        self.linear3 = nn.Linear(self.fc_hidden_units, self.dim_treatments)

        self.linear4 = nn.Linear(self.br_size + self.dim_treatments, self.fc_hidden_units)
        self.elu3 = nn.ELU()
        self.linear5 = nn.Linear(self.fc_hidden_units, self.dim_outcome)

        self.treatment_head_params = ['linear2', 'linear3']


    def build_outcome(self, br, curr_A):
        ''' 
        GY: outcome prediction network
        '''
        # print('br, curr_A shape', br.shape, curr_A.shape) #br [32, 59,10] curr_A [32, 1, 5]
        x = torch.cat((br, curr_A), dim=-1)
        x = self.elu3(self.linear4(x))
        outcome = self.linear5(x)
        return outcome

    def build_br(self, output):
        br = self.elu1(self.linear1(output))
        return br

import numpy as np

def unroll_temporal_data(data_full, observed_nodes_list, window_len, t_step = 1):
    n_samples = data_full.shape[0]
    n_timesteps = data_full.shape[1] # number of data samples
    n_contemporaneous_nodes = data_full.shape[2]
    num_nodes_unrolled = window_len * n_contemporaneous_nodes

    # calculate the starting time index for each unrolled sample
    starts = [time * n_contemporaneous_nodes for time in np.arange(0, n_timesteps + 1 - window_len, t_step)]

    # create unrolled_data 
    data_full_unrolled = np.zeros((n_samples, len(starts), num_nodes_unrolled))
    for i, st in enumerate(starts):
        data_full_unrolled[:,i] = data_full.reshape(n_samples, -1)[:,st:st + num_nodes_unrolled]
    
    # indexes of variables in the unrolled data
    _nodes_sets_list_full = np.reshape(range(num_nodes_unrolled), (window_len, n_contemporaneous_nodes))
    _nodes_sets_list = [_nodes_sets_list_full[i, observed_nodes_list].tolist() for i in range(window_len)]  # observed

    return data_full_unrolled, _nodes_sets_list_full, _nodes_sets_list


class GatedLinearUnit(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Linear(input_dim, output_dim)
    
    def forward(self, inputs):
        return self.linear(inputs) * torch.sigmoid(self.sigmoid(inputs))
    
class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, self.input_dim)
        self.fc2 = nn.Linear(self.input_dim, self.input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.gated_linear_unit = GatedLinearUnit(self.input_dim, self.output_dim)
        self.layer_norm = nn.LayerNorm(self.output_dim)
        self.project = None  # Initialized only if needed in the forward method

    def forward(self, inputs, context_vector = None):
        if context_vector is not None:
            x = torch.cat((inputs, context_vector), dim = -1)
            x = F.elu(self.fc1(x))
        else:
            x = F.elu(self.fc1(inputs))
        x = self.fc2(x)
        x = self.dropout(x)
        if inputs.shape[-1] != self.output_dim:
            if self.project is None:
                self.project = nn.Linear(self.input_dim, self.output_dim)
            inputs = self.project(inputs)
        x = inputs + self.gated_linear_unit(x)
        x = self.layer_norm(x)
        if torch.isnan(x).any():
            print(f"NaN detected in GRN output")
        return x

class VariableSelection(nn.Module):
    def __init__(self, feature, num_variable, output_unit, dropout_rate = 0.2, context_size = 0):
        super().__init__()
        self.feature = 'prev_' + str(feature)
        self.grns = nn.ModuleList([GatedResidualNetwork(output_unit, output_unit, dropout_rate) for _ in range(num_variable)])
        self.grn_concat = GatedResidualNetwork(num_variable * output_unit + context_size, num_variable, dropout_rate)
        self.softmax = nn.Softmax(dim = -1)
        self.embedding = nn.Linear(1, output_unit)

    def forward(self, batch, context_vector = None):
        '''
        Params:
            feature: treatments/covariates/outcomes
        '''
        batch_size, timesteps, num_features = batch.shape
        inputs = [] 
        for i in range(num_features):
            inputs.append(self.embedding(batch[:,:,i].unsqueeze(-1)))
        v = torch.cat(inputs, dim=-1) # (32, 99, 35)
        v = self.grn_concat(v) # (32, 99, 7)
        v = self.softmax(v)
        sparse_weights = v.unsqueeze(-2) # (32, 99, 7, 1)
        # v shape (batch, seq_len, num_feat, 1)
        x = torch.stack([grn(input_tensor) for grn, input_tensor in zip(self.grns, inputs)], dim=-1) # (32, 99, 5, 7)

        combined = sparse_weights * x
        temporal_ctx = torch.sum(combined, dim = -1) # (32, 99, 5)
        return temporal_ctx, sparse_weights # (batch_size, seq_length, output_unit)

class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size, stride = 1):
        super(SeriesDecomposition, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.avg = nn.AvgPool1d(kernel_size = self.kernel_size, stride = stride, padding = 0)
    def _moving_avg(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim = 1)
        x = self.avg(x.permute(0,2,1))
        x = x.permute(0,2,1)
        return x
    def forward(self, x):
        trend = self._moving_avg(x)
        seasonal = x - trend
        return seasonal, trend

