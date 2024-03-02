import torch
import torch.nn as nn

class BasicCNN(nn.Module):
    """ Basic CNN Architecture that was presented in Discussion #6"""
    def __init__(self):
        super(BasicCNN, self).__init__()
        
        # Conv. Block 1
        self.ConvBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=22, out_channels=25, kernel_size=(10, 1), padding=(5, 0)),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(num_features=25),
            nn.Dropout2d(p=0.5)   
        )

        # Conv. Block 2
        self.ConvBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(10, 1), padding=(5, 0)),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(num_features=50),
            nn.Dropout2d(p=0.5)
        )
        
        # Conv. Block 3
        self.ConvBlock3 = nn.Sequential(
            nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(10, 1), padding=(5, 0)),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(num_features=100),
            nn.Dropout2d(p=0.5)
        )

        # Conv. Block 4
        self.ConvBlock4 = nn.Sequential(
            nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(10, 1), padding=(5, 0)),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(num_features=200),
            nn.Dropout2d(p=0.5)
        )

        # Output Layer
        self.Output = nn.Sequential(
            nn.Linear(200*7*1, 4),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.ConvBlock1(x)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.ConvBlock4(x)
        x = x.flatten(start_dim=1,end_dim=3)
        x = self.Output(x)
        return x