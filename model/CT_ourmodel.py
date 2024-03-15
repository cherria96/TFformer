from pytorch_lightning import LightningModule
import torch
from torch import nn
from torch_ema import ExponentialMovingAverage
import torch.optim as optim
from utils_transformer import TransformerMultiInputBlock, AbsolutePositionalEncoding, RelativePositionalEncoding
from utils import BROutcomeHead

'''
수진
- active entries?? 무슨 용도인지 아직 모르겠음. 
- 우선 Input 3개 (X,A,Y) 만 있다고 가정했음. (projection horizon 제외했음)
'''

class CT(LightningModule):
    def __init__(self, ):
        super().__init__()

        self.basic_block_cls = TransformerMultiInputBlock

        # Params for input transformation
        self.dim_A = None # TBD or get from init input
        self.dim_X = None # TBD or get from init input
        self.dim_Y = None # TBD or get from init input
        self.dim_V = None # TBD or get from init input
        self.dim_hidden = None # TBD

        # Parameters for basic block cls
        self.seq_hidden_units = None #TBD
        self.num_heads = None #TBD
        self.head_size = None #TBD
        self.dropout_rate = None #TBD
        self.num_layer = None #TBD

        # Params for poisitional encoding
        # Follow the parameter of ct config file https://github.com/Valentyn1997/CausalTransformer/blob/27b253affa1a1e5190452be487fcbd45093dca00/config/backbone/ct.yaml#L20
        # Type of positional_encoding (abolute/relative) - Appendix D
        self.positional_encoding_absolute = False # use relative cause it is better
        self.max_seq_length = 60 #Originally it's different for each data (cancer sim:60, mimic3_real:60, mimic3_synthetic:100)
        self.positional_encoding_trainable = True
        self.pe_max_relative_position = 15

        self.cpe_max = 18 # TBD
        self.cpe_max_relative_position = None #TBD
        self.cross_positional_encoding_trainable = None #TBD
        
        # Exponential Moving Average
        self.ema = True
        self.beta = 0.99
        if self.ema:
            self.ema_treatment = ExponentialMovingAverage(self.model.parameter(),decay = self.beta)


        sub_args = None
        self.__init__specific(sub_args)

        self.is_augmentation = False
        self.has_vitals = True

        
    """
    (경민)
    우리 한가지 case(multi)인데 init이랑 init specific이 나누어야 할까?
    sub args나 외부 argument쓰는게 헷갈리게 만드는 요인같은데 일단 가능한 거는 init함수에 써볼게!
    """
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
        self.A_input_transformation = nn.Linear(self.dim_A, self.dim_hidden) 
        self.X_input_transformation = nn.Linear(self.dim_X, self.dim_hidden)
        self.Y_input_transformation = nn.Linear(self.dim_Y, self.dim_hidden) 
        self.V_input_transformation = nn.Linear(self.dim_V, self.dim_hidden)
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
        positional encoding k,v가 어디서 정의되야 하는걸까?, 왜 overview에는 안그려져 있지?
        isolate_subnetwork뜻이 뭐지
        -> either 't' 'v' 'o' ''
        e.g. if 't': isolation for the subnetwork dealing with 'treatments'. 
        treatment-related information doesn't interfere with learning of other data types within the model
        """
        self.transformer_blocks = nn.ModuleList([self.basic_block_cls(self.seq_hidden_units, 
                                                                      self.num_heads, self.head_size, 
                                                                      self.seq_hidden_units * 4,self.dropout_rate,
                                                                      self.dropout_rate if sub_args.attn_dropout else 0.0,
                                                                      self_positional_encoding_k=self.self_positional_encoding_k,
                                                                      self_positional_encoding_v=self.self_positional_encoding_v
                                                                      ,n_inputs=self.n_inputs,disable_cross_attention = False,
                                                                      isolate_subnetwork=sub_args.isolate_subnetwork) for _ in range(self.num_layer)])


        # cross positional encoding
        self.cross_positional_encoding = self.cross_positional_encoding_k = self.cross_positional_encoding_v = None
        self.cross_positional_encoding_k = \
            RelativePositionalEncoding(self.cpe_max_relative_position, self.head_size,
                                        self.cross_positional_encoding_trainable, cross_attn = True)
        self.cross_positional_encoding_v = \
            RelativePositionalEncoding(self.cpe_max_relative_position, self.head_size,
                                        self.cross_positional_encoding_trainable, cross_attn = True)

        # dropout
        self.output_dropout = nn.Dropout(self.dropout_rate)
        
        # output layer 
        self.br_head = BROutcomeHead()

    def forward(self, batch):
        '''
        batch: Dict
            batch.keys = [prev_A, X, prev_Y, static_inputs, curr_A, active_entries]
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

        prev_A = batch["prev_A"]
        X = batch["X"] if self.has_vitals else None
        prev_Y = batch["prev_Y"]
        static_features = batch["static inputs"]
        curr_A = batch["curr_A"]
        active_entries = batch['active_entries']

        br = self.build_br(prev_A, X, prev_Y, static_features, active_entries, fixed_split)
        outcome_pred = self.br_head.build_outcome(br, curr_A)
        return br, outcome_pred
    
    def build_br(self, prev_A, X, prev_Y, static_features, active_entries, fixed_split):
        '''
        '''
        active_entries_Y = torch.clone(active_entries)
        active_entries_X = torch.clone(active_entries)

        if fixed_split is not None and self.has_vitals:
            for i in range(len(active_entries)):
                #Masking X in range [fixed_split: ]
                active_entries_X[i, int(fixed_split[i]):, : ] = 0.0
                X[i, int(fixed_split[i]):] = 0.0

        x_t = self.A_input_transformation(prev_A)
        x_o = self.Y_input_transformation(prev_Y)
        x_v = self.X_input_transformation(X) if self.has_vitals else None
        x_s = self.V_input_transformation(static_features.unsqueeze(1))

        for block in self.transformer_blocks:
            if self.self_positional_encoding is not None:
                x_t = x_t + self.self_positional_encoding(x_t)
                x_o = x_o + self.self_positional_encoding(x_o)
                x_v = x_v + self.self_positional_encoding(x_v) if self.has_vitals else None

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
                x = (x_o + x_t + x_v) / 3   
        
        output = self.output_dropout(x)
        br = self.br_head.build_br(output)
        return br


    def training_step(self, batch, batch_idx):
        if self.ema:
            with self.ema_treatment.average_parmeters():
                treatment_pred, outcome_pred, _ = self.model(batch) #model 구조에 따라 바꾸어야 할 듯, prediction part만 쓴 거라서 본래 모델의 성능을 파악하기는 어려울 듯함.
        else:
            outcome_pred, _ = self.model(batch)
        mse_loss = nn.functional.mse_loss(outcome_pred, batch['outputs'], reduce=False)
        self.log(f'train_mse_loss', mse_loss, on_epoch=True, on_step=False, sync_dist=True)
        return mse_loss

    def validation_step(self, batch, batch_idx):
        if self.ema:
            with self.ema_treatment.average_parmeters():
                outcome_pred, _ = self.model(batch) 
        else:
            outcome_pred, _ = self.model(batch) 
        mse_loss = nn.functional.mse_loss(outcome_pred, batch['outputs'], reduce=False)
        self.log(f'vlidation_mse_loss', mse_loss, on_epoch=True, on_step=False, sync_dist=True)
    

    def test_step(self, batch, batch_idx):
        if self.ema:
            with self.ema_treatment.average_parmeters():
                outcome_pred, _ = self.model(batch) 
        else:
            outcome_pred, _ = self.model(batch) 
        mse_loss = nn.functional.mse_loss(outcome_pred, batch['outputs'], reduce=False)
        self.log(f'vlidation_mse_loss', mse_loss, on_epoch=True, on_step=False, sync_dist=True)
    

    def configure_optimizers(self):
        # Follow config of ct https://github.com/Valentyn1997/CausalTransformer/blob/27b253affa1a1e5190452be487fcbd45093dca00/config/backbone/ct.yaml#L22

        lr = 0.01 
        optimizer = optim.Adam(self.model.parameter(), lr = lr) # 어디서 model paramters 받아와야 하지?  
        lr_scheduler = optim.lr_scheduler.ExpontialLR(optimizer,gamma = 0.99)

        return [optimizer], [lr_scheduler]

