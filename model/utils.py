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
        print('br, curr_A shape', br.shape, curr_A.shape) #br [32, 59,10] curr_A [32, 1, 5]
        x = torch.cat((br, curr_A), dim=-1).float()
        x = self.elu3(self.linear4(x))
        outcome = self.linear5(x)
        return outcome

    def build_br(self, output):
        br = self.elu1(self.linear1(output))
        return br




