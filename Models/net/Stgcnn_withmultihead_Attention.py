import torch
import torch.nn as nn
import torch.nn.functional as F
from net.utils.graph import Graph
from net.utils.tgcn import ConvTemporalGraphical

class MultiheadTemporalAttentionPerJoint(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.gamma = nn.Parameter(torch.tensor(0.1))  # facteur de pondération pour la skip connection

    def forward(self, x):
        """
        x : Tensor de forme (N*M, C, T, V)
        Retour : même forme (N*M, C, T, V)
        """
        N_M, C, T, V = x.shape

        # Réorganiser pour traiter chaque joint séparément :
        # On veut appliquer l'attention sur la dimension temporelle (T) pour chaque joint (V)
        x = x.permute(0, 3, 2, 1).contiguous()  # (N*M, V, T, C)
        x = x.view(N_M * V, T, C)               # (N*M*V, T, C)

        # Multihead attention
        attn_out, _ = self.attn(x, x, x)        # (N*M*V, T, C)

        # Skip connection + projection
        out = self.linear(attn_out) + x         # (N*M*V, T, C)
        out = self.gamma * out

        # Revenir à (N*M, C, T, V)
        out = out.view(N_M, V, T, C).permute(0, 3, 2, 1).contiguous()  # (N*M, C, T, V)
        return out
class MultiheadSpatialAttentionPerFrame(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.gamma = nn.Parameter(torch.tensor(0.1))  # facteur de pondération

    def forward(self, x):
        """
        x : Tensor de forme (N*M, C, T, V)
        Retour : même forme (N*M, C, T, V)
        """
        N_M, C, T, V = x.shape

        # Préparer pour attention spatiale par frame
        # Pour chaque t dans T, on applique attention sur les V joints
        x = x.permute(0, 2, 3, 1).contiguous()   # (N*M, T, V, C)
        x = x.view(N_M * T, V, C)                # (N*M*T, V, C)

        # Attention spatiale
        attn_out, _ = self.attn(x, x, x)         # (N*M*T, V, C)

        # Skip connection + projection
        out = self.linear(attn_out) + x          # (N*M*T, V, C)
        out = self.gamma * out

        # Revenir à (N*M, C, T, V)
        out = out.view(N_M, T, V, C).permute(0, 3, 1, 2).contiguous()  # (N*M, C, T, V)
        return out




class Model(nn.Module):
    def __init__(self, in_channels, num_class, graph_args, edge_importance_weighting, max_persons=1, **kwargs):
        super().__init__()
        self.max_persons = max_persons
        self.num_class = num_class
        
        # Load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # Build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 15
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        
        
        self.st_gcn_networks = nn.ModuleList([
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, dropout=0.3),
            st_gcn(64, 64, kernel_size, 1, residual=True,dropout=0.3),
            st_gcn(64, 64, kernel_size, 1,residual=True,dropout=0.3),
            st_gcn(64, 64, kernel_size, 1, residual=True,dropout=0.4),
            st_gcn(64, 128, kernel_size, 2,residual=True, dropout=0.4),
            st_gcn(128, 128, kernel_size, 1,residual=True, dropout=0.4),
            st_gcn(128, 128, kernel_size, 1,residual=True, dropout=0.4),
            st_gcn(128, 256, kernel_size, 2,residual=True, dropout=0.5),
            st_gcn(256, 256, kernel_size, 1,residual=True,dropout=0.5),
            st_gcn(256, 256, kernel_size, 1,residual=True, dropout=0.5),
        ])



        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for _ in range(len(self.st_gcn_networks))
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)
        self.dropout_final = nn.Dropout(0.5)
        # Couche finale CRITIQUE - version garantie
        self.final_conv = nn.Conv2d(256, num_class, kernel_size=1)
        
        self.attn_pool =MultiheadTemporalAttentionPerJoint(embed_dim=256,num_heads=2)

    def forward(self, x):
        N, C, T, V, M = x.size()  # N=batch, C=channels, T=frames, V=joints, M=persons
        
        # Normalisation
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # N, M, V, C, T
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # N, M, C, T, V
        x = x.view(N * M, C, T, V)  # Fusion batch et personnes

        # Forward through ST-GCN blocks
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # Pooling global
        x=self.attn_pool(x) # (N*M, 256, 1, 1)
        x = F.avg_pool2d(x, x.size()[2:])  
        # Classification finale
        x = self.final_conv(x)
        x = self.dropout_final(x)   # (N*M, num_class, 1, 1)
        x = x.view(N, M, -1)  # (N, M, num_class)
        
        return x

class st_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0, residual=True):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A