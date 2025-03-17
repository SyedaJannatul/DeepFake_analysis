import numpy as np
import math 
import torch
import torch.nn as nn
import torch.nn.functional as F

##_____________________Define:MODEL-F & MODEL-G_________________

# Positional Encoding 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(0.1)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,batch_first=True),
            num_layers=num_layers
        )

    def preprocess_latent(self, Z):
        batch_size, channels, height, width = Z.shape # (batch_size, 256, 32, 32)
        seq_len = height * width
        Z = Z.permute(0, 2, 3, 1).reshape(batch_size, seq_len, channels)  # (batch_size, 1024, 256)
        return Z

    def postprocess_latent(self, Z):
        batch_size, seq_len, channels = Z.shape  # (batch_size, 1024, 256)
        height = width = int(math.sqrt(seq_len))
        Z = Z.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)  # (batch_size, 256, 32, 32)
        return Z

    def forward(self, Z):
        Z = self.preprocess_latent(Z)
        Z = self.positional_encoding(Z)
        Z = self.encoder(Z)
        Z = self.postprocess_latent(Z)
        return Z # latent of transformer      

class TransformerDecoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=12, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Enhanced positional encoding
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Multi-layer learnable start tokens
        self.base_start = nn.Parameter(torch.randn(1, 1024, d_model))
        self.start_net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Context-aware transformer decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True            
            ),
            num_layers=num_layers
        )
        
        # Output projection with residual
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.GELU(),
            nn.Linear(d_model*2, d_model))
        
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.base_start, mean=0, std=0.02)

    def preprocess_latent(self, Z):
        # Convert (B, C, H, W) to (B, H*W, C)
        return Z.permute(0, 2, 3, 1).flatten(1, 2)

    def postprocess_latent(self, Z):
        # Convert (B, H*W, C) back to (B, C, H, W)
        B, L, C = Z.shape
        H = W = int(L**0.5)
        return Z.view(B, H, W, C).permute(0, 3, 1, 2)

    def forward(self, Z, Z1_start_tokens=None, teacher_forcing_ratio=0.5):
        # Process input latent
        Z = self.preprocess_latent(Z)
        #Z = self.positional_encoding(Z)
        
        # Generate enhanced start tokens
        B = Z.size(0)
        base_tokens = self.base_start.expand(B, -1, -1)
        processed_start = self.start_net(base_tokens)
        
        # Teacher forcing integration
        if Z1_start_tokens is not None and teacher_forcing_ratio > 0:
            Z1_processed = self.positional_encoding(self.preprocess_latent(Z1_start_tokens))
            
            # Create mixing mask
            mask = torch.rand(B, 1, 1, device=Z.device) < teacher_forcing_ratio
            processed_start = torch.where(mask, Z1_processed, processed_start)

        # Decoder processing with residual
        decoder_input = self.positional_encoding(processed_start)
        outputs = self.decoder(decoder_input, Z)
        outputs = self.output_layer(outputs + decoder_input)
        
        return self.postprocess_latent(outputs)

class DeepfakeToSourceTransformer(nn.Module):
    def __init__(self, d_model=256, encoder_nhead=8, decoder_nhead=8, num_encoder_layers=6, num_decoder_layers=12, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(
            d_model=d_model, 
            nhead=encoder_nhead, 
            num_layers=num_encoder_layers, 
            dim_feedforward=1024, 
            dropout=dropout
        )
        self.decoder = TransformerDecoder(
            d_model=d_model,
            nhead=decoder_nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

    def forward(self, Z, Z1_start_tokens=None, teacher_forcing_ratio=0.5):
        memory = self.encoder(Z)
        Z1 = self.decoder(memory, Z1_start_tokens, teacher_forcing_ratio=teacher_forcing_ratio)
        return Z1
