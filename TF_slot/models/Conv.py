import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import lightning as L
import torchmetrics

# class Conv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
#         super(Conv2d, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU()
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x

# class Conv1d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
#         super(Conv1d, self).__init__()
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x
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
        b, n, d = inputs.shape # (B, b, w*d)
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
class Chomp1d(nn.Module):
    """PyTorch does not offer native support for causal convolutions, so it is implemented (with some inefficiency) by simply using a standard convolution with zero padding on both sides, and chopping off the end of the sequence."""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class FirstBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, groups):
        super(FirstBlock, self).__init__()
        
        # self.target = target
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, groups=groups)

        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.PReLU(n_outputs)
        self.dropout = nn.Dropout(p = 0.3)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout)
        self.relu = nn.PReLU(n_inputs)
        self.init_weights()

    def init_weights(self):
        """Initialize weights"""
        self.conv1.weight.data.normal_(0, 0.1) 
        
    def forward(self, x):
        out = self.net(x)
        return self.relu(out)    

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, groups):
        super(TemporalBlock, self).__init__()
       
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, groups=groups)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.PReLU(n_outputs)
        self.dropout = nn.Dropout(p = 0.3)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout)
        self.relu = nn.PReLU(n_outputs)
        
        self.init_weights()

    def init_weights(self):
        """Initialize weights"""
        self.conv1.weight.data.normal_(0, 0.1) 
        

    def forward(self, x):
        out = self.net(x)
        return self.relu(out+x) #residual connection

class LastBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, groups):
        super(LastBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(n_inputs, n_inputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, groups=groups)
        self.chomp1 = Chomp1d(padding)
        self.net = nn.Sequential(self.conv1, self.chomp1)
        self.linear = nn.Linear(n_inputs, n_outputs)
        self.init_weights()

    def init_weights(self):
        """Initialize weights"""
        self.linear.weight.data.normal_(0, 0.01) 
        
    def forward(self, x):
        out = self.net(x)
        return self.linear(out.transpose(1,2)+x.transpose(1,2)).transpose(1,2) #residual connection
        # return self.linear(out.transpose(1,2)).transpose(1,2) #residual connection




class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs,num_outputs, num_levels, groups, kernel_size=2, dilation_c=2):
        super(TemporalConvNet, self).__init__()
        layers = []
        in_channels = num_inputs
        out_channels = num_outputs
        for l in range(num_levels):
            dilation_size = dilation_c ** l
            if l==0:
                layers += [FirstBlock(in_channels, in_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, groups= groups)]
            elif l==num_levels-1:
                layers+=[LastBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, groups= groups)]
            
            else:
                layers += [TemporalBlock(in_channels, in_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, groups= groups)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNAutoEncoder(L.LightningModule):
    def __init__(self, configs, num_slots, scaler, num_levels=2, kernel_size=2, dilation_c=2):
        super(TCNAutoEncoder, self).__init__()
        
        self.configs = configs

        # config
        self.num_slots = num_slots
        

        # Layers
        self.encoder = TemporalConvNet(num_inputs = configs.enc_in * configs.small_batch_size, num_outputs = configs.d_model * configs.small_batch_size, 
                                       num_levels = num_levels, kernel_size = kernel_size, dilation_c = dilation_c, groups = configs.small_batch_size)
        self.fc1 = nn.Linear(configs.d_model * configs.seq_len,configs.d_model * configs.seq_len//2)
        self.fc2 = nn.Linear(configs.d_model * configs.seq_len//2,configs.d_model * configs.seq_len//4)

        
        self.get_slot = SlotAttention(
            num_slots = self.num_slots, 
            dim = configs.seq_len * configs.d_model, 
            iters = 5, 
            hidden_dim =  configs.d_model
        )
        
        self.decoder = TemporalConvNet(num_inputs = configs.d_model * configs.small_batch_size//4, num_outputs = (configs.c_out) * configs.small_batch_size, 
                                       num_levels = num_levels, kernel_size = kernel_size, dilation_c = dilation_c, groups = configs.small_batch_size)

        metrics = torchmetrics.MetricCollection([torchmetrics.MeanSquaredError(), torchmetrics.MeanAbsoluteError()])
        self.val_metrics = metrics.clone(prefix="Val_")
        self.test_metrics = metrics.clone(prefix="Test_")
        self.scaler = scaler


    
    def _forward(self, batch_x):

        '''
        Propose 2
        ---------------
        x (B, b, w, f)
        input (B, f, w, b) -> encoder -> (B, d, w, b) -> (B, w*b, d) -> slot 
        -> (B, n_s, d) -> (B, n_s, w', b', d) -> (B*n_s, d, w', b') -> decoder
        -> (B*n_s, f+1, w, b) -> (B, n_s, b, w, f+1) -> (B, n_s, b, w, f) & (B, n_s, b, w, 1) -> (B, b, w, f)
        '''
        
        B, b, w, f = batch_x.shape
        enc_input = batch_x.permute(0, 3, 2, 1)
        enc_output = self.encoder(enc_input)
        slot_input = enc_output.permute(0,2,3,1).reshape(B, -1, enc_output.shape[1])
        slot_output = self.get_slot(slot_input)
        slot_output = slot_output.unsqueeze(2).unsqueeze(3)
        slot_output = slot_output.repeat(1,1,w,b,1) # TBU w,b 
        decoder_input = slot_output.permute(0,1,4,2,3).reshape(-1, slot_output.shape[-1], w, b)
        decoder_output = self.decoder(decoder_input)
        decoder_output = decoder_output.permute(0,3,2,1).reshape(B, self.num_slots, b, w, f+1)
        recons, masks = decoder_output.split([f, 1], dim = -1)
        masks = nn.Softmax(dim = 1)
        out_combine = torch.sum(recons * masks, dim = 1) 
        return out_combine
    
    def forward(self, batch_x):
        '''
        Propose 3
        --------------
        x (B, b, w, f)
        input (B, f * b, w) -> encoder(group=b)-> (B, d * b, w) -> (B, b, w*d) -> (B, b, w*d // 4) -> slot 
        -> (B, n_s, w*d//4) -> (B, n_s, b, w*d//4) -> (B*n_s, b*d//4, w) -> decoder (group = b)
        -> (B*n_s, b(f+1), w) -> (B, n_s, b, w, f+1) -> (B, n_s, b, w, f) & (B, n_s, b, w, 1) -> (B, b, w, f)
        '''
        device = next(self.parameters()).device
        B, b, w, f = batch_x.shape
        enc_input = batch_x.permute(0, 3, 2, 1).reshape(B, -1, w)
        enc_output = self.encoder(enc_input)
        slot_input = enc_output.reshape(B, -1, b, w).permute(0,2,3,1).reshape(B, b, -1)
        
        breakpoint()
        slot_input = nn.LayerNorm(slot_input.shape[1:]).to(device)(slot_input)
        slot_input = self.fc1(slot_input)
        slot_input = F.relu(slot_input)
        slot_input = self.fc2(slot_input)

        slot_output = self.get_slot(slot_input)
        slot_output = slot_output.unsqueeze(2)
        slot_output = slot_output.repeat(1,1,b,1)
        decoder_input = slot_output.reshape(B*self.num_slots, -1, w)
        decoder_output = self.decoder(decoder_input)

        decoder_output = decoder_output.reshape(B, self.num_slots, b, f+1, w).permute(0,1,2,4,3)
        recons, masks = decoder_output.split([f, 1], dim = -1)
        masks = nn.Softmax(dim = 1)(masks)
        out_combine = torch.sum(recons * masks, dim = 1) 
        return out_combine[:,:,-self.configs.pred_len:,:]
    

    def shared_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        outputs = self(batch_x)
        f_dim = -1 if self.configs.variate == "mu" else 0
        batch_y = batch_y[:, :,-self.configs.pred_len :, f_dim:]
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





