
import torch
import torch.nn as nn
import torch.nn.functional as F
import math    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=30):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, frames, features)
        return x + self.pe[:x.size(1)].transpose(0, 1)
  
class SpatialCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # (B*T, C, J, P) -> (B*T, 32, J, P)
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),           # (B*T, 64, J, P)
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, out_channels, kernel_size=3, padding=1), # (B*T, out_channels, J, P)
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        B, C, T, J, P = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, J, P)
        x = x.view(B * T, C, J, P)                 # (B*T, C, J, P)
        x = self.encoder(x)                        # (B*T, F, J, P)
        x = x.view(B, T, -1, J, P)                 # (B, T, F, J, P)
        x = x.permute(0, 2, 1, 3, 4)               # (B, F, T, J, P)
        return x
class MultiPersonContextLSTM(nn.Module):
    def __init__(self, 
                 input_features=3,  # e.g. x, y, angle
                 num_joints=18,
                 num_classes=2,
                 hidden_dim=256,
                 num_layers=2,
                 num_heads=4,
                 max_persons=3,
                 dropout=0.3):
        super().__init__()

        self.num_joints = num_joints
        self.max_persons = max_persons
        self.joint_embed_dim = hidden_dim // 2

        
        self.spatial_cnn = SpatialCNN(in_channels=input_features, out_channels=input_features)

        
        self.joint_embed = nn.Sequential(
            nn.Linear(input_features, self.joint_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        
        self.pos_encoder = PositionalEncoding(self.joint_embed_dim * num_joints)

        # 3. LSTM
        self.lstm = nn.LSTM(
            input_size=self.joint_embed_dim * num_joints,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 4. Normalisation
        self.layernorm = nn.LayerNorm(hidden_dim)

        # 5. Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )

        # 6. Classificateur
        self.person_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        # x: (B, C, T, J, P)
        x = self.spatial_cnn(x)  # (B, C, T, J, P)
        B, C, T, J, P = x.size()
        all_person_outputs = []

        for p in range(P):
            x_p = x[..., p].permute(0, 2, 3, 1)  # (B,T,J,C)
            x_p = self.joint_embed(x_p)         # (B,T,J,E)
            x_p = x_p.view(B, T, -1)            # (B,T,J*E)
            x_p = self.pos_encoder(x_p)

            lstm_out, _ = self.lstm(x_p)
            lstm_out = self.layernorm(lstm_out)

            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            pooled = attn_out.mean(dim=1)
            out_p = self.person_classifier(pooled)
            all_person_outputs.append(out_p.unsqueeze(1))

        return torch.cat(all_person_outputs, dim=1)  # (B, P, C)
