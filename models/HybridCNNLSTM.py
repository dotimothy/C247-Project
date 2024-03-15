import torch
import torch.nn as nn
import numpy as np 
from keras.utils import to_categorical

class HybridCNNLSTM(nn.Module):
    def __init__(self,chunk_size=400):
        """ DeepConvNet LSTM: Optimized from Discussion #7 """
        super(HybridCNNLSTM, self).__init__()
        # Metadata
        self.name = "HybridCNNLSTM"
        
        # Conv. block 1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=22, out_channels=25, kernel_size=(10, 1), padding=(5,0)),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(25),
            nn.Dropout(0.6)
        )
        
        # Conv. block 2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(10, 1), padding=(5,0)),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(50),
            nn.Dropout(0.6)
        )
        
        # Conv. block 3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(10, 1), padding=(5,0)),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(100),
            nn.Dropout(0.6)
        )
        
        # Conv. block 4
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(10, 1), padding=(5,0)),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(200),
            nn.Dropout(0.6)
        )
        
        # FC + LSTM layers
        # self.num_fc = self.determine_fc_size(chunk_size)
        # self.fc = nn.Sequential(
        #     nn.Linear(self.num_fc, 40),
        #     nn.ReLU(),
        #     # nn.Linear(40, 1),
        #     # nn.ReLU()
        # )

        self.lstm_input = self.determine_lstm_input(chunk_size)
        self.lstm = nn.LSTM(input_size=self.lstm_input, hidden_size=10, num_layers=2, dropout=0.4, batch_first=True, bidirectional=True)
        #self.lstm_dropout = nn.Dropout(0.4)
      
        # Output layer with Softmax activation
        self.output_layer = nn.Sequential(
            #nn.Linear(20,4)
            nn.Linear(200*20, 4),
            nn.Softmax(dim=1)
        )

    # def determine_fc_size(self,chunk_size):
    #     with torch.no_grad():
    #         x = torch.zeros(2,22,chunk_size,1)
    #         x = self.conv_block1(x)
    #         x = self.conv_block2(x)
    #         x = self.conv_block3(x)
    #         x = self.conv_block4(x)
    #         return 200*x.shape[2]*1

    def determine_lstm_input(self,chunk_size):
        with torch.no_grad():
            x = torch.zeros(2,22,chunk_size,1)
            x = self.conv_block1(x)
            x = self.conv_block2(x)
            x = self.conv_block3(x)
            x = self.conv_block4(x)
            return x.shape[2]
        
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        # Flatten the output
        #x = x.view(x.size(0), -1)
        #print("Flatten: ",x.shape)
        
        # FC layer
        #x = self.fc(x)
        #print("FC: ", x.shape )
        
        # Reshape for LSTM
        x = x.flatten(start_dim=2,end_dim=3)
        #x = x.view(-1,200,1)

        # LSTM layer
        #print(x.shape)
        x, _ = self.lstm(x)
        #x = self.lstm_dropout(x)
        #print(x.shape)
        
        # Output layer
        #x = x[:,-1.:] # only use last output
        x = x.flatten(start_dim=1,end_dim=2) # use all outputs
        x = self.output_layer(x)
        
        return x

def train_data_prep(X,y,sub_sample,average,noise,chunk_size=800):
    
    total_X = None
    total_y = None
    
    # Trimming the data (sample,22,1000) -> (sample,22,800)
    X = X[:,:,0:chunk_size]
    
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

def test_valid_data_prep(X,chunk_size=800):
    
    total_X = None
    
    
    # Trimming the data (sample,22,1000) -> (sample,22,800)
    X = X[:,:,0:chunk_size]
    
    # Maxpooling the data (sample,22,800) -> (sample,22,800/sub_sample)
    X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, 2), axis=3)
    
    
    total_X = X_max
    
    return total_X

def DatasetLoaders(data_dir='./project_data/project',batch_size=256,augment=False,data_leak=False,chunk_size=500):
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

    if(augment): 
      if(data_leak): # Old Way where Val and Train were augmented
        X_train_valid_prep,y_train_valid_prep = train_data_prep(X_train_valid,y_train_valid,2,2,True,chunk_size)
        
        # First generating the training and validation indices using random splitting
        ind_valid = np.random.choice(8460, 1000, replace=False)
        ind_train = np.array(list(set(range(8460)).difference(set(ind_valid))))
        
        # Creating the training and validation sets using the generated indices
        (x_train, x_valid) = X_train_valid_prep[ind_train], X_train_valid_prep[ind_valid] 
        (y_train, y_valid) = y_train_valid_prep[ind_train], y_train_valid_prep[ind_valid]
      else:
        # First generating the training and validation indices using random splitting
        ind_valid = np.random.choice(2115, 500, replace=False)
        ind_train = np.array(list(set(range(2115)).difference(set(ind_valid))))
  
        # Splitting
        (x_train_prep, x_valid) = X_train_valid[ind_train], X_train_valid[ind_valid]
        (y_train_prep, y_valid) = y_train_valid[ind_train], y_train_valid[ind_valid]
  
        # Apply Augmentation to Training Set Only
        x_train, y_train = train_data_prep(x_train_prep, y_train_prep,2,2,True,chunk_size)
        
        ## Preprocessing the other Subsets
        x_valid = test_valid_data_prep(x_valid,chunk_size)
      X_test_prep = test_valid_data_prep(X_test,chunk_size)  
    else:
      ## Simple Truncation of Time-Series
      X_train_valid_prep = X_train_valid[:,:,0:chunk_size]
      X_test_prep = X_test[:,:,0:chunk_size]
      
      ## Random splitting and reshaping the data
      # First generating the training and validation indices using random splitting
      ind_valid = np.random.choice(2115, 500, replace=False)
      ind_train = np.array(list(set(range(2115)).difference(set(ind_valid))))
      
      # Creating the training and validation sets using the generated indices
      (x_train, x_valid) = X_train_valid_prep[ind_train], X_train_valid_prep[ind_valid] 
      (y_train, y_valid) = y_train_valid[ind_train], y_train_valid[ind_valid]
  
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