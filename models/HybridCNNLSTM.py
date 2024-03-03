import torch
import torch.nn as nn
import numpy as np 
from keras.utils import to_categorical

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
            nn.Linear(1000, 40),
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

def train_data_prep(X,y,sub_sample,average,noise):
    total_X = None
    total_y = None
    
    # Trimming the data (sample,22,1000) -> (sample,22,800)
    X = X[:,:,0:800]
    
    # Maxpooling the data (sample,22,800) -> (sample,22,800/sub_sample)
    X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, sub_sample), axis=3)
    
    
    total_X = X_max
    total_y = y
    
    # Averaging + noise 
    X_average = np.mean(X.reshape(X.shape[0], X.shape[1], -1, average),axis=3)
    X_average = X_average + np.random.normal(0.0, 0.5, X_average.shape)
    
    total_X = np.vstack((total_X, X_average))
    total_y = np.hstack((total_y, y))
    
    # Subsampling
    
    for i in range(sub_sample):
        
        X_subsample = X[:, :, i::sub_sample] + \
                            (np.random.normal(0.0, 0.5, X[:, :,i::sub_sample].shape) if noise else 0.0)
            
        total_X = np.vstack((total_X, X_subsample))
        total_y = np.hstack((total_y, y))
        
    
    return total_X,total_y

def test_data_prep(X):
    
    total_X = None
    
    # Trimming the data (sample,22,1000) -> (sample,22,800)
    X = X[:,:,0:800]
    
    # Maxpooling the data (sample,22,800) -> (sample,22,800/sub_sample)
    X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, 2), axis=3)
    
    
    total_X = X_max
    
    return total_X
        
def DatasetLoaders(data_dir='./project_data/project',batch_size=256):
    """ Function to Load in the Datasets for Preprocessing """
    ## Loading the dataset
    X_test = np.load(f"{data_dir}/X_test.npy")
    y_test = np.load(f"{data_dir}/y_test.npy")
    person_train_valid = np.load(f"{data_dir}/person_train_valid.npy")
    X_train_valid = np.load(f"{data_dir}/X_train_valid.npy")
    y_train_valid = np.load(f"{data_dir}/y_train_valid.npy")
    person_test = np.load(f"{data_dir}/person_test.npy")

    ## Adjusting the labels so that 
    
    # Cue onset left - 0
    # Cue onset right - 1
    # Cue onset foot - 2
    # Cue onset tongue - 3
    
    y_train_valid -= 769
    y_test -= 769

    ## Preprocessing the dataset
    X_train_valid_prep,y_train_valid_prep = train_data_prep(X_train_valid,y_train_valid,2,2,True)
    X_test_prep = test_data_prep(X_test) 
    
    ## Random splitting and reshaping the data
    
    # First generating the training and validation indices using random splitting
    ind_valid = np.random.choice(8460, 1000, replace=False)
    ind_train = np.array(list(set(range(8460)).difference(set(ind_valid))))
    
    # Creating the training and validation sets using the generated indices
    (x_train, x_valid) = X_train_valid_prep[ind_train], X_train_valid_prep[ind_valid] 
    (y_train, y_valid) = y_train_valid_prep[ind_train], y_train_valid_prep[ind_valid]
    
    
    # Converting the labels to categorical variables for multiclass classification
    y_train = to_categorical(y_train, 4)
    y_valid = to_categorical(y_valid, 4)
    y_test = to_categorical(y_test, 4)
    
    
    # Adding width of the segment to be 1
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], x_train.shape[2], 1)
    x_test = X_test_prep.reshape(X_test_prep.shape[0], X_test_prep.shape[1], X_test_prep.shape[2], 1)

    # Creating Data Tensors & Datasets
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_valid_tensor = torch.tensor(x_valid, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    train_data = torch.utils.data.TensorDataset(x_train_tensor,y_train_tensor)
    valid_data = torch.utils.data.TensorDataset(x_valid_tensor,y_valid_tensor)
    test_data = torch.utils.data.TensorDataset(x_test_tensor,y_test_tensor)
    train_loader = torch.utils.data.DataLoader(train_data,shuffle=True,batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(valid_data,shuffle=False,batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data,shuffle=False,batch_size=batch_size)

    return train_data,valid_data,test_data,train_loader,val_loader,test_loader