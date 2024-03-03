import torch
import torch.nn as nn
import numpy as np
from keras.utils import to_categorical

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

        self.model_name = "BasicCNN"

    def forward(self, x):
        x = self.ConvBlock1(x)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.ConvBlock4(x)
        x = x.flatten(start_dim=1,end_dim=3)
        x = self.Output(x)
        return x

def DatasetLoaders(data_dir='./project_data/project',batch_size=256,augment=False):
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
    X_train_valid_prep = X_train_valid[:,:,0:500]
    X_test_prep = X_test[:,:,0:500]
    
    ## Random splitting and reshaping the data
    # First generating the training and validation indices using random splitting
    ind_valid = np.random.choice(2115, 500, replace=False)
    ind_train = np.array(list(set(range(2115)).difference(set(ind_valid))))
    
    # Creating the training and validation sets using the generated indices
    (x_train, x_valid) = X_train_valid_prep[ind_train], X_train_valid_prep[ind_valid] 
    (y_train, y_valid) = y_train_valid[ind_train], y_train_valid[ind_valid]

    if(augment): # Apply Augmentation to Training Set Only
      y_train_og = y_train
      slide = 10
      stride = 5
      for s in range(slide):
        X_train_aug = X_train_valid[ind_train,:,s*stride:(s*stride)+500] # Adjsut window of samples
        y_train_aug = y_train_og # Same class label regardless of window
        x_train = np.vstack((x_train,X_train_aug))
        y_train = np.hstack((y_train,y_train_aug))
  
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