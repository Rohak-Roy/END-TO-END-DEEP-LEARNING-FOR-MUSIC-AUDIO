import torch
from torch import nn 
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(self, length, stride): 
        super(Model, self).__init__()

        self.additional_strided_conv = nn.Conv1d(
            in_channels=1,
            out_channels= 1,
            kernel_size= length,
            stride= stride
        )

        self.conv1 = nn.Conv1d(
            in_channels= 1,
            out_channels= 32,
            kernel_size= 8,
        )

        self.pool1 = nn.MaxPool1d(kernel_size= 4, stride=1)

        self.conv2 = nn.Conv1d(
            in_channels= 32,
            out_channels= 32,
            kernel_size= 8,
        )

        self.pool2 = nn.MaxPool1d(kernel_size= 4, stride=1) 

        self.fc1 = nn.Linear(
            in_features= 32,
            out_features= 100
        )

        self.fc2 = nn.Linear(
            in_features= 100,
            out_features= 50
        )

        self.avg_pool_layer = nn.AvgPool1d(kernel_size=10)

        self.initialise_layer(self.additional_strided_conv)
        self.initialise_layer(self.conv1)
        self.initialise_layer(self.conv2)
        self.initialise_layer(self.fc1)
        self.initialise_layer(self.fc2)

    def forward(self, x):
        B, S, C, _ = x.shape
        x = torch.flatten(x, start_dim=0, end_dim=1)            # [B, 10, 1, 34950] --> [10B, 1, 34950]
        x = F.relu(self.additional_strided_conv(x))  
        x = F.relu(self.conv1(x))                   
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)                     
        pool2_B, pool2_C, pool2_F = x.shape
        pool2_avg_over_F = nn.AvgPool1d(kernel_size= pool2_F)
        x = pool2_avg_over_F(x)                                 # [10B, 32, F] --> [10B, 32, 1]
        x = x.reshape((pool2_B, pool2_C))                       # [10B, 32, 1] --> [10B, 32]
        x = F.relu(self.fc1(x))                                 # [10B, 32]    --> [10B, 100]
        x = self.fc2(x)                                         # [10B, 100]   --> [10B, 50]
        x = x.reshape((B, S, -1))                               # [10B, 50]    --> [B, 10, 50] 
        x = x.transpose(1, 2)                                   # [B, 10, 50]  --> [B, 50, 10]  
        x = self.avg_pool_layer(x)                              # [B, 50, 10]  --> [B, 50, 1]
        x = x.transpose(1, 2)                                   # [B, 50, 1]   --> [B, 1, 50]
        x = torch.flatten(x, start_dim=1)                       # [B, 50]
        x = torch.sigmoid(x)                                   
        return x                                                
    
    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)
