import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os

PAD_TOKEN_ID = 0

# ----------------------
# 3) Positional Encoding
# ----------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension to `pe`
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: Tensor of shape (B, S, d_model)
        """
        seq_len = x.size(1)  # Sequence length is in dim 1 for (B, S, d_model)
        if seq_len > self.pe.size(1):
            raise ValueError(f"Input sequence length ({seq_len}) exceeds max_len ({self.pe.size(1)}).")
        
        # Add positional encodings
        x = x + self.pe[:, :seq_len, :]  # Slice positional encodings to match input length
        return self.dropout(x)

# ---------------------------
# 4) Encoder & Decoder Stacks
# ---------------------------

class EncoderStack(nn.Module):
    def __init__(self, d_model, d_ffn, nhead, num_layers, dropout):
        # Inherit the constructor of PyTorch Module
        super(EncoderStack, self).__init__()

        # Define the encoder layer and stack
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, dim_feedforward=d_ffn, nhead=nhead, dropout=dropout)
        self.layers = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, src):
        output = self.layers(src)
        return output

class DecoderStack(nn.Module):
    def __init__(self, d_model, d_ffn, nhead, num_layers, dropout):
        # Inherit the constructor of PyTorch Module
        super(DecoderStack, self).__init__()
        
        # Define the decoder layer and stack
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, dim_feedforward=d_ffn, nhead=nhead, dropout=dropout)
        self.layers = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

    def causal_mask(self, n):
        mask = torch.triu(torch.ones(n, n), diagonal=1)  # Upper triangular mask
        mask = mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, 0)  
        return mask.float()

    def forward(self, tgt, memory, maskQ = False):
        # Causal mask, tgt and src (i.e. encoder memory)
        tgt_seq_len = tgt.size(1)
        tgt_mask = self.causal_mask(tgt_seq_len).to(tgt.device)

        return self.layers(tgt, memory, tgt_mask = tgt_mask, tgt_is_causal = True)


# ----------------------
# 5) Core Model that embodies the space of integrals (this is what we train)
# ----------------------
class KernelModel(nn.Module):
    def __init__(self, vocab_size, d_model=96, d_ffn=96, nhead=8, encoder_layers=4, decoder_layers=4, dropout=0.1):
        # Causal mask, tgt and src (i.e. encoder memory)
        super(KernelModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN_ID)

        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout=dropout)

        self.encoder = EncoderStack(d_model, d_ffn, nhead, encoder_layers, dropout)
        self.decoder = DecoderStack(d_model, d_ffn, nhead, decoder_layers, dropout)

        self.fc_out = nn.Linear(d_model, vocab_size)

        self.d_model = d_model
        self.d_fnn = d_ffn
        self.nhead = nhead
        self.encoder_layers = decoder_layers
        self.decoder_layers = decoder_layers
        self.vocab_size = vocab_size



    def forward(self, src, tgt):
        # Embed + positional encode
        src_emb = self.embedding(src)         # (B,S) ==> (B, S, d_model)
        src_emb = self.pos_encoder(src_emb)   # (B, S, d_model)

        tgt_emb = self.embedding(tgt)         # (B, T, d_model)
        tgt_emb = self.pos_decoder(tgt_emb)   # (B, T, d_model)

        # Encode source sequence
        memory = self.encoder(src_emb)  # => (B, S, d_model)
        
        output = self.decoder(tgt_emb, memory)  # => (B, T, d_model)
        return self.fc_out(output)    # => (B, T, vocab_size)