import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import lightning as L
import torchmetrics
import torch.nn.functional as F



class Model(nn.Module):
    """
    Vanilla Transformer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        
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
    def shared_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        outputs = self(batch_x, batch_y, batch_x_mark, batch_y_mark)
        f_dim = -1 if self.configs.variate == "mu" else 0
        batch_y = batch_y[:, -self.model.pred_len :, f_dim:]
        return outputs, batch_y

    def training_step(self, batch, batch_idx):
        outputs, batch_y = self.shared_step(batch, batch_idx)
        loss = self.loss(outputs, batch_y)
        self.log("Train_Loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

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