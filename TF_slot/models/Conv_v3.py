import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import lightning as L
import torchmetrics

from sklearn.preprocessing import StandardScaler

class Chomp1d(nn.Module):
    """PyTorch does not offer native support for causal convolutions, so it is implemented (with some inefficiency) by simply using a standard convolution with zero padding on both sides, and chopping off the end of the sequence."""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class FirstBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(FirstBlock, self).__init__()
        
        # self.target = target
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, groups=n_outputs)

        self.chomp1 = Chomp1d(padding)
        self.net = nn.Sequential(self.conv1, self.chomp1)      
        self.relu = nn.PReLU(n_inputs)
        self.init_weights()

    def init_weights(self):
        """Initialize weights"""
        self.conv1.weight.data.normal_(0, 0.1) 
        
    def forward(self, x):
        out = self.net(x)
        return self.relu(out)    

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(TemporalBlock, self).__init__()
       
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, groups=n_outputs)
        self.chomp1 = Chomp1d(padding)
        self.net = nn.Sequential(self.conv1, self.chomp1)
        self.relu = nn.PReLU(n_outputs)
        self.init_weights()

    def init_weights(self):
        """Initialize weights"""
        self.conv1.weight.data.normal_(0, 0.1) 
        

    def forward(self, x):
        out = self.net(x)
        return self.relu(out+x) #residual connection

class LastBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(LastBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(n_inputs, n_inputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, groups=n_outputs)
        self.chomp1 = Chomp1d(padding)
        self.net = nn.Sequential(self.conv1, self.chomp1)
        self.linear = nn.Linear(n_inputs, n_outputs)
        self.init_weights()

    def init_weights(self):
        """Initialize weights"""
        self.linear.weight.data.normal_(0, 0.01) 
        
    def forward(self, x):
        out = self.net(x)
        # return self.linear(out.transpose(1,2)+x.transpose(1,2)).transpose(1,2) #residual connection

        return self.linear(out.transpose(1,2)).transpose(1,2) #residual connection

class DepthwiseNet(nn.Module):
    def __init__(self, num_inputs,num_outputs, num_levels, kernel_size=2, dilation_c=2,dec=False):
        super(DepthwiseNet, self).__init__()
        layers = []
        in_channels = num_inputs
        out_channels = num_outputs
        for l in range(num_levels):
            dilation_size = dilation_c ** l
            if l==0:
                layers += [FirstBlock(in_channels, in_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size)]
            elif l==num_levels-1:
                if dec == True:
                    layers+=[LastBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                        padding=(kernel_size-1) * dilation_size+1)]
                else:
                    layers+=[LastBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size)]
            
            else:
                layers += [TemporalBlock(in_channels, in_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)



class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.rand(1, 1, dim))

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots = None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_sigma.expand(b, n_s, -1)
        slots = torch.normal(mu, sigma)

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))

        return slots
    

class DepthConv1d(L.LightningModule):
    def __init__(self, config, num_levels, kernel_size, dilation_c, scaler):
        super(DepthConv1d, self).__init__()
        
        self.configs = config

        feature_len = config.enc_in-1
        input_len = config.seq_len # + config.label_len
        pred_len = config.pred_len
        num_slot = config.num_clusters
        small_batch_size = config.small_batch_size
        # self.target=target
        self.dwn = DepthwiseNet(input_len*feature_len, input_len*feature_len, num_levels, kernel_size=kernel_size, dilation_c=dilation_c)
        self.dec = DepthwiseNet(small_batch_size, small_batch_size, num_levels, kernel_size=kernel_size, dilation_c=dilation_c,dec=True)

        # self.fs_attention = th.nn.Parameter(self._attention.data)
        self.num_slot = num_slot
        self.pred_len = pred_len

        self.get_slot = SlotAttention(
            num_slots = num_slot,
            dim = feature_len*input_len,   # same size as input
            iters = 3,
            hidden_dim=128,   # hidden dimension size of slot attention
        )
        
        metrics = torchmetrics.MetricCollection([torchmetrics.MeanSquaredError(), torchmetrics.MeanAbsoluteError()])
        self.val_metrics = metrics.clone(prefix="Val_")
        self.test_metrics = metrics.clone(prefix="Test_")
        self.scaler = scaler

    def init_weights(self):
        self.pointwise.weight.data.normal_(0, 0.1)       
        
    def forward(self, x):
        # y1=self.dwn(x*F.softmax(self.fs_attention, dim=0))
        B, b, w, f = x.shape
        # print("Shape:", B,b,w,f)
        y1=self.dwn(x.reshape((B,b,-1)).permute(0,2,1))   # shape [B,b,wf] -> [B, wf, b]
        # slot attention
        y1 = y1.permute(0,2,1) # [B, b, wf]
        slot = self.get_slot(y1)    # [B, num_slot, wf]

        ####### normalization per slot ########
        mean_slot = slot.mean(-1, keepdim=True).detach()
        slot_gap = slot - mean_slot
        std_slot = torch.sqrt(torch.var(slot, dim=-1, keepdim=True, unbiased=False) + 1e-5).detach()
        slot = slot_gap / std_slot
        #######################################

        slot = slot.reshape((-1,1,w*f)).repeat(1,b,1)   # [B*num_slot, wf]# [B*num_slot, b, wf]

        # decoder
        dec_out = self.dec(slot)   #  [B*num_slot, b, wf+1]
        dec_out = dec_out.reshape((B, self.num_slot,b, -1))
        out, masks = dec_out.split([w*f,1],dim=-1)   
        # out: [B, num_slot, b, wf]
        # mask: [B, num_slot, b, 1]
        masks = nn.Softmax(dim=1)(masks)
        out_val = out*masks   # [B, num_slot, b, wf]
        ####### denormalization per slot ########
        std_slot = std_slot.unsqueeze(-1)#.unsqueeze(-1)
        mean_slot = mean_slot.unsqueeze(-1)#.unsqueeze(-1)
        out_val = out_val * std_slot + mean_slot
        #########################################
        out_combine = torch.sum(out_val,dim=1)   # [B, b, wf]
        out_combine = out_combine.reshape((B,b,-1, f))

        return out_combine[:,:,-self.configs.pred_len:,:]
    
    def shared_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        outputs = self(batch_x)
        f_dim = -1 if self.configs.variate == "mu" else 0
        batch_y = batch_y[:, :,-self.pred_len :, f_dim:]
        return outputs, batch_y
    
    def training_step(self, batch, batch_idx):
        outputs, batch_y = self.shared_step(batch, batch_idx)
        loss = self.loss(outputs, batch_y)
        self.log("Train_Loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.configs.lr)
        if self.configs.scheduler == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
        elif self.configs.scheduler == "two_step_exp":

            def two_step_exp(epoch):
                if epoch % 4 == 2:
                    return 0.5
                if epoch % 4 == 0:
                    return 0.2
                return 1.0

            scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=[two_step_exp])
        else:
            raise RuntimeError("The scheduler {self.configs.lr_scheduler} is not implemented.")
        return [optimizer], [scheduler]

    def loss(self, outputs, targets, **kwargs):
        if self.configs.loss == "mse":
            return F.mse_loss(outputs, targets)
        raise RuntimeError("The loss function {self.configs.loss} is not implemented.")

    def validation_step(self, batch, batch_idx):
        outputs, batch_y = self.shared_step(batch, batch_idx)
        loss = self.loss(outputs, batch_y)
        self.log("Val_Loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if self.configs.inverse_scaling and self.scaler is not None:
            outputs = self.scaler.inverse_transform(outputs)
            batch_y = self.scaler.inverse_transform(batch_y)
        self.val_metrics(outputs, batch_y)
        self.log_dict(self.val_metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"outputs": outputs, "targets": batch_y}

    def test_step(self, batch, batch_idx):
        outputs, batch_y = self.shared_step(batch, batch_idx)
        if self.configs.inverse_scaling and self.scaler is not None:
            outputs = self.scaler.inverse_transform(outputs)
            batch_y = self.scaler.inverse_transform(batch_y)
        self.test_metrics(outputs, batch_y)
        self.log_dict(self.test_metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"outputs": outputs, "targets": batch_y}



