
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


class Transformer(nn.Module):
    def __init__(self, input_size=1024, encoder_layers=6, attention_heads=8, more_residuals=False, max_length=None,
                 pos_embed='simple', epsilon=1e-5, weight_init=None):
        super(Transformer, self).__init__()
        self.input_size = input_size
        self.max_length = max_length
        # Positional embeddings
        if self.max_length:
            self.pos_embed_type = pos_embed
            if self.pos_embed_type == "simple":
                self.pos_embed = nn.Embedding(self.max_length, self.input_size)
            elif self.pos_embed_type == "attention":
                self.pos_embed = torch.zeros(self.max_length, self.input_size)
                for pos in range(self.max_length):
                    for i in np.arange(0, self.input_size, 2):
                        self.pos_embed[pos, i] = np.sin(pos / (10000 ** ((2 * i) / self.input_size)))
                        self.pos_embed[pos, i + 1] = np.cos(pos / 10000 ** ((2 * (i + 1)) / self.input_size))

                    else:
                        self.max_length = None
                # Add the residual connection to the output of the encoder
        self.more_residuals = more_residuals
        self.dropout = nn.Dropout(0.5)
        self.layer_norm = torch.nn.LayerNorm(self.input_size, epsilon)

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=attention_heads,
                                                                    dim_feedforward=self.input_size, dropout=0.1,
                                                                    activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.transformer_encoder_layer,
                                                         num_layers=encoder_layers, norm=self.layer_norm)

        self.k1 = nn.Linear(in_features=self.input_size, out_features=self.input_size)
        self.k2 = nn.Linear(in_features=self.input_size, out_features=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        if weight_init:
            if weight_init.lower() in ["he", "kaiming"]:
                for i in np.arange(self.transformer_encoder.num_layers):
                    init.kaiming_uniform_(self.transformer_encoder.layers[i].linear1.weight)
                    init.kaiming_uniform_(self.transformer_encoder.layers[i].linear2.weight)
                init.kaiming_uniform_(self.k1.weight)
                init.kaiming_uniform_(self.k2.weight)
            elif weight_init.lower() == "xavier":
                for i in np.arange(self.transformer_encoder.num_layers):
                    init.xavier_uniform_(self.transformer_encoder.layers[i].linear1.weight)
                    init.xavier_uniform_(self.transformer_encoder.layers[i].linear2.weight)
                init.xavier_uniform_(self.k1.weight)
                init.xavier_uniform_(self.k2.weight)

    def forward(self, x):
        """
        Input
           x: (seq_len, batch_size, input_size)
        Output:
           y: (seq_len, batch_size, 1)
        """
        seq_len, batch_size, input_size = x.shape
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, input_size)
        if self.max_length is not None:
            assert self.max_length >= seq_len, "input sequence has higher length than max_length"
            if self.pos_embed_type == "simple":
                pos_tensor = torch.arange(seq_len).repeat(1, batch_size).view([batch_size, seq_len]).to(x.device)
                x += self.pos_embed(pos_tensor)
            elif self.pos_embed_type == 'attention':
                x += self.pos_embed[:seq_len, :].repeat(1, batch_size).view(batch_size, seq_len, input_size).to(
                    x.device)

        x = x.permute(1, 0, 2)
        encoder_out = self.transformer_encoder.forward(x)
        if self.more_residuals:
            encoder_out += x
        y = self.k1(encoder_out)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.layer_norm(y)
        y = self.k2(y)
        y = self.sigmoid(y)
        return y
