import sys
sys.path.append('/Users/sujinchoi/Desktop/TF_slot')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim
import lightning as L
import torchmetrics

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
from layers.SlotAttention import SlotAttention
from layers.Preprocess import rolling_average
import pdb 




class Model(nn.Module):
    '''
    Vanilla Transformer
    '''

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        
        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model,
                                           configs.embed, configs.freq, 
                                           configs.dropout)
        
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model,
                                           configs.embed, configs.freq, 
                                           configs.dropout)
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout = configs.dropout,
                                      output_attention = configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout = configs.dropout,
                    activation = configs.activation
                    ) for l in range(configs.e_layers)
            ],
            norm_layer = torch.nn.LayerNorm(configs.d_model)
        )
        self.configs = configs    
        # KMeans
    
        self.do_kmeans = configs.kmeans
        self.do_slotattention = configs.slotattention

        self.pca = PCA(n_components= configs.n_components)
        self.kmeans = KMeans(n_clusters = configs.num_clusters, n_init = 'auto')

        self.slot_attention = SlotAttention(num_slots= 10, dim= configs.seq_len*configs.d_model, iters = 10, hidden_dim= 128)

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout = configs.dropout, output_attention = False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout = configs.dropout, output_attention = False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout = configs.dropout,
                    activation = configs.activation,
                    ) for l in range(configs.d_layers)
            ],
            norm_layer = torch.nn.LayerNorm(configs.d_model),
        )
        self.projection = nn.Linear(configs.d_model, configs.c_out + 1, bias = True)

        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask = None, dec_self_mask = None, dec_enc_mask = None, 
                window_size = 7):
        device = next(self.parameters()).device

        # Normalization 
        # x_raw = x_enc.clone().detach()
        # mean_enc = x_enc.mean(1, keepdim=True).detach() # B x 1 x E
        # x_enc = x_enc - mean_enc
        # std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach() # B x 1 x E
        # x_enc = x_enc / std_enc
        # x_dec_new = torch.cat([x_enc[:, -self.label_len: , :], torch.zeros_like(x_dec[:, -self.pred_len:, :])], dim=1).to(x_enc.device).clone()

        # tau = self.tau_learner(x_raw, std_enc).exp() # b, 1
        # delta = self.delta_learner(x_raw, mean_enc)  # b, w
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask = enc_self_mask) 
        # enc_out = enc_out.view(B, b, w, d)
        if self.do_kmeans:
            enc_rolling = rolling_average(enc_out, window_size = window_size) # (b, w', d_model)
            enc_rolling = enc_rolling.reshape(enc_out.shape[0], -1) # (b, w'* d_model)
            enc_pca = self.pca.fit_transform(enc_rolling.cpu().detach().numpy()) # (b, n_components)
            cluster_labels = self.kmeans.fit_predict(enc_pca) # (b,)
            cluster_labels = torch.tensor(cluster_labels)
            
            dec_out = []
            for label in torch.unique(cluster_labels):
                cluster_indices = torch.where(cluster_labels == label)[0]
                cluster_pca = enc_pca[cluster_indices.numpy()] # (b', n_components)
                enc_one_cluster = self.pca.inverse_transform(cluster_pca) # (b', w'*d_model)
                enc_one_cluster = enc_one_cluster.reshape(enc_one_cluster.shape[0], -1, self.configs.d_model) # (b', w', d_model)
                enc_one_cluster = torch.FloatTensor(enc_one_cluster).to(device)
                
                #normalization
                b, w, d = enc_one_cluster.shape
                enc_one_cluster = enc_one_cluster.reshape(b, -1)
                mean_enc = enc_one_cluster.mean(1, keepdim=True).detach() # b', 1
                enc_one_cluster = enc_one_cluster - mean_enc
                std_enc = torch.sqrt(torch.var(enc_one_cluster, dim=1, keepdim=True, unbiased=False) + 1e-5).detach() # b', 1
                enc_one_cluster = enc_one_cluster / std_enc
                enc_one_cluster = enc_one_cluster.reshape(b,w,d)

                x_dec_one_cluster = x_dec[cluster_indices]
                x_dec_mark_one_cluster = x_mark_dec[cluster_indices]
                x_dec_one_cluster = self.dec_embedding(x_dec_one_cluster, x_dec_mark_one_cluster)
                x_dec_one_cluster = torch.cat([enc_one_cluster[:, -self.configs.label_len: , :], torch.zeros_like(x_dec_one_cluster[:, -self.pred_len:, :])], dim=1).to(enc_one_cluster.device).clone()
                dec_out_one_cluster = self.decoder(x_dec_one_cluster, enc_one_cluster, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
                dec_out_one_cluster = self.projection(dec_out_one_cluster)
                recon, mask = dec_out_one_cluster.split([self.configs.c_out, 1], dim = -1)
                mask = nn.Softmax(dim =1)(mask)
                dec_out_one_cluster = recon * mask
                dec_out_one_cluster = dec_out_one_cluster.reshape(b, -1)
                dec_out_one_cluster = dec_out_one_cluster * std_enc + mean_enc
                dec_out_one_cluster = dec_out_one_cluster.reshape(b, w, -1)

                dec_out.append(dec_out_one_cluster)
            dec_out = torch.cat(dec_out, dim = 0)
        
        elif self.do_slotattention:
            # enc_out.shape (b,w,d)
            breakpoint()
            self.slot_attention(enc_out)




        else:
            dec_out = self.dec_embedding(x_dec, x_mark_dec)
            dec_out = self.decoder(dec_out, enc_out, x_mask = dec_self_mask, cross_mask = dec_enc_mask)
            dec_out = self.projection(dec_out)


        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]
    
class Config:
    def __init__(self,
                path,
                data_type,
                data,  # "synthetic", "AD"
                seq_len= 60,
                label_len= 60,
                pred_len= 30,
                variate= 'm',
                target= None,
                scale= True,
                is_timeencoded= False,
                random_state= 42,
                output_attention = False,
                enc_in = 8,
                d_model = 512,
                embed = 'fixed',
                freq = 'd',
                dropout = 0.05,
                dec_in = 8,
                factor = 5,
                n_heads = 8,
                d_ff = 2048,
                activation = 'gelu',
                e_layers = 3,
                n_components = 10,
                num_clusters = 5,
                d_layers = 3,
                c_out = 8,
                batch_size= 32,
                epoch= 30,
                lr= 0.0005,
                loss= 'mse',
                scheduler= 'exponential',
                inverse_scaling = True,
                kmeans = True,
                slotattention = False,
                num_workers = 0,
                small_batch_size = None,
                small_stride = 1,


                 ):

        self.path = path
        self.data_type = data_type
        self.data = data
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.variate = variate
        self.target = target
        self.scale = scale
        self.is_timeencoded = is_timeencoded
        self.random_state = random_state
        self.output_attention = output_attention
        self.enc_in = enc_in
        self.d_model = d_model
        self.embed = embed
        self.freq = freq
        self.dropout = dropout
        self.dec_in = dec_in
        self.factor = factor
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.activation = activation
        self.e_layers = e_layers
        self.n_components = n_components
        self.num_clusters = num_clusters
        self.d_layers = d_layers
        self.c_out = c_out
        self.batch_size = batch_size
        self.epoch = epoch
        self.lr = lr
        self.loss = loss
        self.scheduler = scheduler
        self.inverse_scaling = inverse_scaling
        self.kmeans = kmeans
        self.num_workers = num_workers
        self.slotattention = slotattention
        self.small_batch_size = small_batch_size
        self.small_stride = small_stride

    def to_dict(self):
        return {key: value for key, value in self.__dict__.items()}


        
class TimeSeriesForecasting(L.LightningModule):
    def __init__(
        self,
        configs,
        scaler
    ):
        super(TimeSeriesForecasting, self).__init__()
        self.configs = configs
        self.model = Model(configs)
        self.save_hyperparameters(ignore=["model"])
        metrics = torchmetrics.MetricCollection([torchmetrics.MeanSquaredError(), torchmetrics.MeanAbsoluteError()])
        self.val_metrics = metrics.clone(prefix="Val_")
        self.test_metrics = metrics.clone(prefix="Test_")
        self.scaler = scaler

    def forward(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        decoder_input = torch.zeros((batch_y.size(0), self.configs.pred_len, batch_y.size(-1))).type_as(batch_y)
        decoder_input = torch.cat([batch_y[:, : self.configs.label_len, :], decoder_input], dim=1)
        outputs = self.model(batch_x, batch_x_mark, decoder_input, batch_y_mark)
        if self.model.output_attention:
            outputs = outputs[0]
        return outputs
    def shared_step(self, batch):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        outputs = self(batch_x, batch_y, batch_x_mark, batch_y_mark)
        f_dim = -1 if self.configs.variate == "mu" else 0
        batch_y = batch_y[:, -self.model.pred_len :, f_dim:]
        return outputs, batch_y

    def training_step(self, batch, batch_idx):
        if batch[0].ndim == 4:
            Batch_size = batch[0].size(0)
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            loss_all = []
            for i in range(Batch_size):
                small_batch = (batch_x[i], batch_y[i], batch_x_mark[i], batch_y_mark[i])
                outputs, true = self.shared_step(small_batch, )
                loss = self.loss(outputs, true)
                loss_all.append(loss)
            loss = sum(loss_all) / Batch_size
        else:
            outputs, batch_y = self.shared_step(batch, )
            loss = self.loss(outputs, batch_y)

        self.log("Train_Loss_mse", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if batch[0].ndim == 4:
            Batch_size = batch[0].size(0)
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            loss_all = []
            for i in range(Batch_size):
                small_batch = (batch_x[i], batch_y[i], batch_x_mark[i], batch_y_mark[i])
                outputs, true = self.shared_step(small_batch, )
                if self.configs.inverse_scaling and self.scaler is not None:
                    outputs = self.scaler.inverse_transform(outputs)
                    true = self.scaler.inverse_transform(true)
                # self.val_metrics(outputs, true)
                loss = torch.sqrt(self.loss(outputs, true))
                
                loss_all.append(loss)
            loss = sum(loss_all) / Batch_size
            self.log("Val_Loss_rmse", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        else:
            outputs, batch_y = self.shared_step(batch)
            loss = self.loss(outputs, batch_y)
            self.log("Val_Loss_mse", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            if self.configs.inverse_scaling and self.scaler is not None:
                outputs = self.scaler.inverse_transform(outputs)
                batch_y = self.scaler.inverse_transform(batch_y)
            self.val_metrics(outputs, batch_y)
            self.log_dict(self.val_metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"outputs": outputs, "targets": batch_y}

    def test_step(self, batch, batch_idx):
        if batch[0].ndim == 4:
            Batch_size = batch[0].size(0)
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            loss_all = []
            for i in range(Batch_size):
                small_batch = (batch_x[i], batch_y[i], batch_x_mark[i], batch_y_mark[i])
                outputs, true = self.shared_step(small_batch, )
                if self.configs.inverse_scaling and self.scaler is not None:
                    outputs = self.scaler.inverse_transform(outputs)
                    true = self.scaler.inverse_transform(true)
                # self.val_metrics(outputs, true)
                loss = torch.sqrt(self.loss(outputs, true))
                
                loss_all.append(loss)
            loss = sum(loss_all) / Batch_size
            self.log("test_Loss_rmse", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        else:
            outputs, batch_y = self.shared_step(batch)
            if self.configs.inverse_scaling and self.scaler is not None:
                outputs = self.scaler.inverse_transform(outputs)
                batch_y = self.scaler.inverse_transform(batch_y)
            self.test_metrics(outputs, batch_y)
            self.log_dict(self.test_metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"outputs": outputs, "targets": batch_y}


    # def on_fit_start(self):
    #     if self.configs.inverse_scaling and self.scaler is not None:
    #         if self.scaler.device == torch.device("cpu"):
    #             self.scaler.to(self.device)

    # def on_test_start(self):
    #     if self.configs.inverse_scaling and self.scaler is not None:
    #         if self.scaler.device == torch.device("cpu"):
    #             self.scaler.to(self.device)

    def loss(self, outputs, targets, **kwargs):
        if self.configs.loss == "mse":
            return F.mse_loss(outputs, targets)
        raise RuntimeError("The loss function {self.configs.loss} is not implemented.")

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