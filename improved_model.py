import torch
from torch import nn 
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(self, length, stride): 
        super(Model, self).__init__()

        self.additional_strided_conv = nn.Conv1d(
            in_channels=1,
            out_channels= 128,
            kernel_size= length,
            stride= stride
        )

        self.initialise_layer(self.additional_strided_conv)        

        self.additional_strided_conv_bn = nn.BatchNorm1d(128)

        self.conv1 = nn.Conv1d(
            in_channels= 128,
            out_channels= 256,
            kernel_size= 8,
        )

        self.initialise_layer(self.conv1)       

        self.conv1_bn = nn.BatchNorm1d(256)

        self.pool1 = nn.MaxPool1d(kernel_size= 4, stride=1)

        self.conv2 = nn.Conv1d(
            in_channels= 256,
            out_channels= 512,
            kernel_size= 8,
        )

        self.initialise_layer(self.conv2)

        self.conv2_bn = nn.BatchNorm1d(512)

        self.pool2 = nn.MaxPool1d(kernel_size= 4, stride=1) 

        self.conv3 = nn.Conv1d(
            in_channels= 512,
            out_channels = 512,
            kernel_size=8
        )

        self.initialise_layer(self.conv3)

        self.conv3_bn = nn.BatchNorm1d(512)

        self.pool3 = nn.MaxPool1d(kernel_size = 4, stride=1)

        self.fc1 = nn.Linear(
            in_features= 512,
            out_features= 100
        )

        self.initialise_layer(self.fc1)

        self.fc2 = nn.Linear(
            in_features= 100,
            out_features= 50
        )

        self.initialise_layer(self.fc2)
        
        self.avg_pool_layer = nn.AvgPool1d(kernel_size=10)

        self.dropOut = nn.Dropout(0.5)

    def forward(self, x):
        B, S, C, _ = x.shape
        x = torch.flatten(x, start_dim=0, end_dim=1) # [B, 10, 1, 34950] --> [10B, 1, 34950]
        x = F.relu(self.additional_strided_conv_bn(self.additional_strided_conv(x)))
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = self.pool2(x) 
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = self.dropOut(x)
        x = self.pool3(x)                
        pool2_B, pool2_C, pool2_F = x.shape
        pool2_avg_over_F = nn.AvgPool1d(kernel_size= pool2_F)
        x = pool2_avg_over_F(x)                 
        x = x.reshape((pool2_B, pool2_C))      
        x = F.relu(self.fc1(x))                 
        x = self.fc2(x)                         
        x = x.reshape((B, S, -1))               
        x = x.transpose(1, 2)                   
        x = self.avg_pool_layer(x)              
        x = x.transpose(1, 2)                  
        x = torch.flatten(x, start_dim=1)      
        x = torch.sigmoid(x)                
        return x
    
    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)
