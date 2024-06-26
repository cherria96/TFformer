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


