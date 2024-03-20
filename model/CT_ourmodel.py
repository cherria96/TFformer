from pytorch_lightning import LightningModule
import torch
from torch import nn
from torch_ema import ExponentialMovingAverage
import torch.optim as optim
from model.utils_transformer import TransformerMultiInputBlock, AbsolutePositionalEncoding, RelativePositionalEncoding
from model.utils import BROutcomeHead
import numpy as np
import pdb
'''
수진
- active entries?? 무슨 용도인지 아직 모르겠음. 
- 우선 Input 3개 (X,A,Y) 만 있다고 가정했음. (projection horizon 제외했음)
'''

class CT(LightningModule):
    def __init__(self, dim_A=None, dim_X = None, dim_Y = None, dim_V = None,
                 seq_hidden_units = 10, num_heads = 2, head_size = None, dropout_rate = 0.2, 
                  num_layer = 1):
        super().__init__()

        self.basic_block_cls = TransformerMultiInputBlock

        # Params for input transformation
        self.dim_A = dim_A # equal to the treatment dimension in original code
        self.dim_X = dim_X 
        self.dim_Y = dim_Y # equal to the outcome dimension in original code
        self.dim_V = dim_V 

        # Parameters for basic block cls
        self.seq_hidden_units = seq_hidden_units #10
        self.num_heads = num_heads # 2
        self.head_size = seq_hidden_units // num_heads 
        self.dropout_rate = dropout_rate # range from 0.1 to 0.5
        self.num_layer = num_layer #1
        self.br_size = 10 # relate to input size
        self.fc_hidden_units = 5 # relate to the size of balanced representation
        self.alpha =0.01

        # Params for poisitional encoding
        # Follow the parameter of ct config file https://github.com/Valentyn1997/CausalTransformer/blob/27b253affa1a1e5190452be487fcbd45093dca00/config/backbone/ct.yaml#L20
        # Type of positional_encoding (abolute/relative) - Appendix D
        self.positional_encoding_absolute = False # use relative cause it is better
        self.max_seq_length = 60 #Originally it's different for each data (cancer sim:60, mimic3_real:60, mimic3_synthetic:100)
        self.positional_encoding_trainable = True
        self.pe_max_relative_position = 15

        self.cpe_max = 18 
        
        # Exponential Moving Average
        self.ema = True
        self.beta = 0.99
        if self.ema:
            self.ema_treatment = ExponentialMovingAverage(self.parameters(),decay = self.beta)


        self.__init__specific()
        self.training = True
        self.is_augmentation = False
        self.has_vitals = False

        
    """
    (경민)
    우리 한가지 case(multi)인데 init이랑 init specific이 나누어야 할까?
    sub args나 외부 argument쓰는게 헷갈리게 만드는 요인같은데 일단 가능한 거는 init함수에 써볼게!
    """
    def __init__specific(self):
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
        # super(CT, self).__init__specific(sub_args)


        # input transformation
        '''
        We need to take these weights to interpret attention matrix
        '''
        self.A_input_transformation = nn.Linear(self.dim_A, self.seq_hidden_units) 
        self.X_input_transformation = nn.Linear(self.dim_X, self.seq_hidden_units)
        self.Y_input_transformation = nn.Linear(self.dim_Y, self.seq_hidden_units) 
        self.V_input_transformation = nn.Linear(self.dim_V, self.seq_hidden_units)
        self.n_inputs = 3

        # Init of positional encoding https://github.com/Valentyn1997/CausalTransformer/blob/27b253affa1a1e5190452be487fcbd45093dca00/src/models/edct.py#L78
        # Use relative positionalencoding only

        self.self_positional_encoding = self.self_positional_encoding_k = self.self_positional_encoding_v = None
        
        self.self_positional_encoding_k = \
            RelativePositionalEncoding(self.pe_max_relative_position, self.head_size,
                                        self.positional_encoding_trainable)
        self.self_positional_encoding_v = \
            RelativePositionalEncoding(self.pe_max_relative_position, self.head_size,
                                        self.positional_encoding_trainable)

        # transformer blocks 
        """
        (경민)
        isolate_subnetwork뜻이 뭐지
        """
        self.transformer_blocks = nn.ModuleList([self.basic_block_cls(self.seq_hidden_units, 
                                                                      self.num_heads, self.head_size, 
                                                                      self.seq_hidden_units * 4,self.dropout_rate,
                                                                      self.dropout_rate ,
                                                                      self_positional_encoding_k=self.self_positional_encoding_k,
                                                                      self_positional_encoding_v=self.self_positional_encoding_v
                                                                      ,n_inputs=self.n_inputs,disable_cross_attention = False,
                                                                      isolate_subnetwork='_') for _ in range(self.num_layer)])


        # dropout
        self.output_dropout = nn.Dropout(self.dropout_rate)
        
        # output layer 
        self.br_head = BROutcomeHead(self.seq_hidden_units, self.br_size, self.fc_hidden_units, self.dim_A, self.dim_Y,self.alpha)

    def forward(self, batch):
        '''
        batch: Dict
            batch.keys = [prev_treatments, X, prev_outputs, static_features, current_treatments, active_entries]
        fixed split : tells the model up to which point in the sequence the data is considered as observed and from which point the data is to be predicted
            model can use real data up to the 'fixed_split' point and then switch to using its predictions or masked values beyond this point
        active entries: (binary tensor) indicates the active or valid entries in the sequence data fro each instance in the batch
            since not all sequences have the same length, batch['active entries'] serves as a mask to differentiate between real data points and padded points
            to ensure that computations only consider the valid parts of each sequence 

        '''
        fixed_split = batch['future_past_split'] if 'future_past_split' in batch else None 
        # Data augmentation of the training data (under 3 conditions: training phase, augmentation flag, presence of Vitals)
        if self.training and self.is_augmentation and self.has_vitals: # Mini-batch augmentation with masking
            assert fixed_split is None
            # twice the size of active entries: for both the original and the augmented copies of the batch
            fixed_split = torch.empty((2 * len(batch['active_entries']),)).type_as(batch['active_entries']) 
            for i, seq_len in enumerate(batch['active entries'].sum(1).int()):
                fixed_split[i] = seq_len #original batch
                fixed_split[len(batch['active_entries']) + i] = torch.randint(0, int(seq_len) + 1, (1,)).item() #augmented batch

            for (k,v) in batch.items():
                batch[k] = torch.cat((v,v), dim = 0)

        prev_treatments = batch["prev_treatments"]
        vitals = batch["vitals"] if self.has_vitals else None
        prev_outputs = batch["prev_outputs"]
        static_features = batch["static_features"]
        current_treatments = batch["current_treatments"]
        active_entries = batch['active_entries']

        #print('pevA, prevY , vitals, static feature shape:',prev_treatments.shape,prev_outputs.shape, vitals.shape, static_features.shape)
        #-> torch.Size([32, 60, 5]) torch.Size([32, 60, 1]) torch.Size([32, 60, 10]) torch.Size([32, 3])
        br = self.build_br(prev_treatments, vitals, prev_outputs, static_features, active_entries, fixed_split)
        #print("br, curr A shape",br.shape, current_treatments.shape)
        #->torch.Size([8, 60, 10]) torch.Size([8, 60, 5])
        outcome_pred = self.br_head.build_outcome(br, current_treatments)
        return br, outcome_pred
    
    def build_br(self, prev_treatments, X, prev_outputs, static_features, active_entries, fixed_split):
        active_entries_Y = torch.clone(active_entries)
        active_entries_X = torch.clone(active_entries)

        if fixed_split is not None and self.has_vitals:
            for i in range(len(active_entries)):
                #Masking X in range [fixed_split: ]
                active_entries_X[i, int(fixed_split[i]):, : ] = 0.0
                X[i, int(fixed_split[i]):] = 0.0

        x_t = self.A_input_transformation(prev_treatments)
        x_o = self.Y_input_transformation(prev_outputs)
        x_v = self.X_input_transformation(X) if self.has_vitals else None
        x_s = self.V_input_transformation(static_features.unsqueeze(1))
        #print('xt, xo, xv shape before:', x_t.shape, x_o.shape, x_v.shape)#(32,60,10) (32,60,10) (32,60,10)

        for block in self.transformer_blocks:
            if self.self_positional_encoding is not None:
                x_t = x_t + self.self_positional_encoding(x_t)
                x_o = x_o + self.self_positional_encoding(x_o)
                x_v = x_v + self.self_positional_encoding(x_v) if self.has_vitals else None
            # print('xt, xo, xv shape:', x_t.shape, x_o.shape, x_v.shape) #(32,60,10) (32,60,10) (32,60,10)
            if self.has_vitals:
                x_t, x_o, x_v = block((x_t, x_o, x_v), x_s, active_entries_Y, active_entries_X)
            else:
                x_t, x_o = block((x_t, x_o), x_s, active_entries_Y)
        if not self.has_vitals:
            x = (x_o + x_t) / 2
        else:
            if fixed_split is not None: # Test seq data
                x = torch.empty_like(x_o)
                for i in range(len(active_entries)):
                    # Masking X in range [fixed_split:]
                    m = int(fixed_split[i])
                    x[i, :m] = (x_o[i, :m] + x_t[i, :m] + x_v[i, :m]) / 3
                    x[i, m:] = (x_o[i, :m] + x_t[i, :m]) / 2
            else: # Train data has always X
                print("shape x_o, x_t, x_v", x_o.shape, x_t.shape, x_v.shape)
                x = (x_o + x_t + x_v) / 3   
        
        output = self.output_dropout(x)
        br = self.br_head.build_br(output)
        return br


    def training_step(self, batch, batch_idx):
        if self.ema:
            with self.ema_treatment.average_parameters():
                _, outcome_pred= self(batch) 
        else:
            _, outcome_pred= self(batch)
        # print('outcome shape, prediction and gt', outcome_pred.shape,batch['outputs'].shape)
        # torch.Size([32, 60, 10]) torch.Size([32, 60, 10])
        mse_loss = nn.functional.mse_loss(outcome_pred, batch['outputs'], reduce=False)
        self.log(f'train_mse_loss', mse_loss.mean(), on_epoch=True, on_step=False, sync_dist=True) 
        # 왜 우리 prediction 결과과 (25,60, 10)이지? 기존 모델은 .mean()안해도 되는데
        # parameter 탓인가?
        return mse_loss.mean()

    def validation_step(self, batch, batch_idx):
        if self.ema:
            with self.ema_treatment.average_parameters():
                outcome_pred, _ = self(batch) 
        else:
            outcome_pred, _ = self(batch) 
        mse_loss = nn.functional.mse_loss(outcome_pred, batch['outputs'], reduce=False)
        self.log(f'validation_mse_loss', mse_loss.mean(), on_epoch=True, on_step=False, sync_dist=True)
    

    def test_step(self, batch, batch_idx):
        if self.ema:
            with self.ema_treatment.average_parameters():
                outcome_pred, _ = self(batch) 
        else:
            outcome_pred, _ = self(batch) 
        mse_loss = nn.functional.mse_loss(outcome_pred, batch['outputs'], reduce=False)
        self.log(f'validation_mse_loss', mse_loss.mean(), on_epoch=True, on_step=False, sync_dist=True)
    

    def configure_optimizers(self):
        # Follow config of ct https://github.com/Valentyn1997/CausalTransformer/blob/27b253affa1a1e5190452be487fcbd45093dca00/config/backbone/ct.yaml#L22

        lr = 0.01 
        optimizer = optim.Adam(self.parameters(), lr = lr)
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma = 0.99)

        return [optimizer], [lr_scheduler]


