from pytorch_lightning import LightningModule
import torch
from torch import nn
from utils import TransformerMultiInputblock, BROutcomeHead
'''
수진
- active entries?? 무슨 용도인지 아직 모르겠음. 
- 우선 Input 3개 (X,A,Y) 만 있다고 가정했음. (projection horizon 제외했음)
'''

class CT(LightningModule):
    def __init__(self, ):
        super().__init__()

        self.basic_block_cls = TransformerMultiInputblock
        sub_args = None
        self.__init__specific(sub_args)
    
    def __init__specific(self, sub_args):
        """
        Initialization of specific sub-network (only multi)
        Args:
            sub_args: sub-network hyperparameters
        Configs:
            dim_hidden
            dim_A : treatments
            dim_X : vitals
            dim_Y : outputs
            dim_V : static inputs
        """
        super(CT, self).__init__specific(sub_args)


        # input transformation
        '''
        We need to take these weights to interpret attention matrix
        '''
        self.A_input_transformation = nn.Linear(dim_A, dim_hidden) 
        self.X_input_transformation = nn.Linear(dim_X, dim_hidden)
        self.Y_input_transformation = nn.Linear(dim_Y, dim_hidden) 
        self.V_input_transformation = nn.Linear(dim_V, dim_hidden)
        self.n_inputs = 3

        # transformer blocks 
        self.transformer_blocks = nn.ModuleList(
            [self.basic_block_cls()]

        )

        # output layer 
        self.br_head = BROutcomeHead()

    def forward(self, batch):
        '''
        batch.items = [prev_A, X, prev_Y, static_inputs, curr_A, active_entries]
        '''
        prev_A = batch["prev_A"]
        X = batch["X"]
        prev_Y = batch["prev_Y"]
        static_features = batch["static inputs"]
        curr_A = batch["curr_A"]
        active_entries = batch['active_entries']

        br = self.build_br(prev_A, X, prev_Y, static_features, active_entries)
        outcome_pred = self.br_head.build_outcome(br, curr_A)
        return br, outcome_pred
    
    def build_br(self, prev_A, X, prev_Y, static_features, active_entries):
        '''
        Required: define self_positional_encoding, output_dropout
        -> src/edct.py
        '''
        active_entries_Y = torch.clone(active_entries)
        active_entries_X = torch.clone(active_entries)

        x_t = self.A_input_transformation(prev_A)
        x_o = self.Y_input_transformation(prev_Y)
        x_v = self.X_input_transformation(X) 
        x_s = self.V_input_transformation(static_features.unsqueeze(1))

        for block in self.transformer_blocks:
            x_t = x_t + self.self_positional_encoding(x_t)
            x_o = x_o + self.self_positional_encoding(x_o)
            x_v = x_v + self.self_positional_encoding(x_v) 

            x_t, x_o, x_v = block((x_t, x_o, x_v), x_s, active_entries_Y, active_entries_X)

            x = (x_o + x_t + x_v) / 3
        
        output = self.output_dropout(x)
        br = self.br_head.build_br(output)
        return br


            



    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        pass


    
    

        

    
    

    