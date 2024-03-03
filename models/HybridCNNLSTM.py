import torch
import torch.nn as nn

class HybridCNNLSTM(nn.Module):
    def __init__(self):
        super(HybridCNNLSTM, self).__init__()
        
        # Conv. block 1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=22, out_channels=25, kernel_size=(5, 5), padding=2),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(25),
            nn.Dropout(0.6)
        )
        
        # Conv. block 2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(5, 5), padding=2),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(50),
            nn.Dropout(0.6)
        )
        
        # Conv. block 3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(5, 5), padding=2),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(100),
            nn.Dropout(0.6)
        )
        
        # Conv. block 4
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(5, 5), padding=2),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(200),
            nn.Dropout(0.6)
        )
        
        # FC + LSTM layers
        self.fc = nn.Sequential(
            nn.Linear(1400, 40),
            nn.ReLU(),
            # nn.Linear(40, 1),
            # nn.ReLU()
        )
        
        self.lstm = nn.LSTM(input_size=1, hidden_size=10, num_layers=1, dropout=0.4, batch_first=True)
        
        # Output layer with Softmax activation
        self.output_layer = nn.Sequential(
            nn.Linear(10, 4),
            nn.Softmax(dim=1)
        )

        self.model_name = "HybridCNNLSTM"
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        # print("Flatten: ",x.shape)
        
        # FC layer
        x = self.fc(x)
        # print("FC: ", x.shape )
        
        # Reshape for LSTM
        x = x.view(-1, 40, 1)

        # LSTM layer
        x, _ = self.lstm(x)
        
        # Output layer
        x = self.output_layer(x[:, -1, :])
        
        return x