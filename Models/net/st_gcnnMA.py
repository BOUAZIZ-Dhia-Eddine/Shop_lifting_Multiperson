import torch
import torch.nn as nn
import torch.nn.functional as F
from net.utils.graph import Graph
from net.utils.tgcn import ConvTemporalGraphical

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
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        
        
        self.st_gcn_networks = nn.ModuleList([
            # Bloc 1 - Couches initiales
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, dropout=0.3),
            st_gcn(64, 64, kernel_size, 1, dropout=0.3),
            st_gcn(64, 64, kernel_size, 1,dropout=0.3),
            st_gcn(64, 64, kernel_size, 1, dropout=0.4),
            st_gcn(64, 128, kernel_size, 2, dropout=0.4),
            st_gcn(128, 128, kernel_size, 1, dropout=0.4),
            st_gcn(128, 128, kernel_size, 1, dropout=0.4),
            st_gcn(128, 256, kernel_size, 2, dropout=0.5),
            st_gcn(256, 256, kernel_size, 1,dropout=0.5),
            st_gcn(256, 256, kernel_size, 1, dropout=0.5),
        ])
        '''            st_gcn(in_channels, 64, kernel_size, 1, residual=False, dropout=0.3),
            st_gcn(64, 64, kernel_size, 1, dropout=0.3),
            st_gcn(64, 64, kernel_size, 1,dropout=0.3),
            st_gcn(64, 64, kernel_size, 1, dropout=0.4),
            st_gcn(64, 128, kernel_size, 2, dropout=0.4),
            st_gcn(128, 128, kernel_size, 1, dropout=0.4),
            st_gcn(128, 128, kernel_size, 1, dropout=0.4),
            st_gcn(128, 256, kernel_size, 2, dropout=0.5),
            st_gcn(256, 256, kernel_size, 1,dropout=0.5),
            st_gcn(256, 256, kernel_size, 1, dropout=0.5),
                        st_gcn(in_channels, 32, kernel_size, stride=1, residual=False, dropout=0.4),                     
            st_gcn(32, 64, kernel_size, stride=1, residual=True, dropout=0.4),                           

            # Bloc 2 - Downsampling + régularisation
            st_gcn(64, 128, kernel_size, stride=2, residual=True, dropout=0.4),  
            st_gcn(128, 128, kernel_size, stride=1, residual=True, dropout=0.4),   

            # Bloc 3 - Profondeur supplémentaire
            st_gcn(128, 256, kernel_size, stride=2, residual=True, dropout=0.5),  
            st_gcn(256, 256, kernel_size, stride=1, residual=True, dropout=0.5), '''


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
        
        self.pool = nn.AdaptiveAvgPool2d(1)

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
        #x = self.pool(x)  # (N*M, 256, 1, 1)
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