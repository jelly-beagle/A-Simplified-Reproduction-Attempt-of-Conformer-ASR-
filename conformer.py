import torch
import torch.nn as nn
import torch.nn.functional as F
from subsampling import RepVGGSEInput


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class ConformerBlock(nn.Module):
    def __init__(self, d_model, nhead=4, dim_feedforward=512):
        super(ConformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )

        self.conv_norm = nn.LayerNorm(d_model)
        self.conv_module = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)

        ffn_out = self.feed_forward(x)
        x = self.norm2(x + ffn_out)

        residual = x
        x = self.conv_norm(x)  # 在 [B, T, 256] 形状下做 Norm
        x = x.transpose(1, 2)  # 转为 [B, 256, T] 进入 Conv1d
        x = self.conv_module(x)
        x = x.transpose(1, 2)  # 转回 [B, T, 256]

        return residual + x


class ConformerEncoder(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=256, num_layers=4, deploy=False):
        super(ConformerEncoder, self).__init__()
        self.subsampling = RepVGGSEInput(in_channels=1, hidden_dim=hidden_dim, deploy=deploy)
        self.pos_encoding = PositionalEncoding(hidden_dim)
        self.layers = nn.ModuleList([
            ConformerBlock(d_model=hidden_dim) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.subsampling(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)

    def switch_to_deploy(self):
        self.subsampling.switch_to_deploy()