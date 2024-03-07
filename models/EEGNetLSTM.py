import torch
import torch.nn as nn
import numpy as np
from keras.utils import to_categorical

# Courtesy of the TorchEEG Repository: https://github.com/torcheeg

# Helper Class for Constrained Conv2D
class Conv2dWithConstraint(nn.Conv2d):
  def __init__(self, *args, max_norm: int = 1, **kwargs):
    self.max_norm = max_norm
    super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
    return super(Conv2dWithConstraint, self).forward(x)
    

class EEGNetLSTM(nn.Module):
  """
  A compact convolutional neural network (EEGNet) with an LSTM. For more details, please refer to the following information.

  - Paper: Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
  - URL: https://arxiv.org/abs/1611.08024
  - Related Project: https://github.com/braindecode/braindecode/tree/master/braindecode

  Below is a recommended suite for use in emotion recognition tasks:

  .. code-block:: python

    dataset = DEAPDataset(io_path=f'./deap',
          root_path='./data_preprocessed_python',
          online_transform=transforms.Compose([
            transforms.To2d()
            transforms.ToTensor(),
          ]),
          label_transform=transforms.Compose([
            transforms.Select('valence'),
            transforms.Binary(5.0),
          ]))
    model = EEGNet(chunk_size=128,
             num_electrodes=32,
             dropout=0.5,
             kernel_1=64,
             kernel_2=16,
             F1=8,
             F2=16,
             D=2,
             num_classes=2)

  Args:
    chunk_size (int): Number of data points included in each EEG chunk, i.e., :math:`T` in the paper. (default: :obj:`151`)
    num_electrodes (int): The number of electrodes, i.e., :math:`C` in the paper. (default: :obj:`60`)
    F1 (int): The filter number of block 1, i.e., :math:`F_1` in the paper. (default: :obj:`8`)
    F2 (int): The filter number of block 2, i.e., :math:`F_2` in the paper. (default: :obj:`16`)
    D (int): The depth multiplier (number of spatial filters), i.e., :math:`D` in the paper. (default: :obj:`2`)
    num_classes (int): The number of classes to predict, i.e., :math:`N` in the paper. (default: :obj:`2`)
    kernel_1 (int): The filter size of block 1. (default: :obj:`64`)
    kernel_2 (int): The filter size of block 2. (default: :obj:`64`)
    dropout (float): Probability of an element to be zeroed in the dropout layers. (default: :obj:`0.25`)
  """
  def __init__(self,
         chunk_size: int = 151,
         num_electrodes: int = 60,
         F1: int = 8,
         F2: int = 16,
         D: int = 2,
         num_classes: int = 2,
         kernel_1: int = 64,
         kernel_2: int = 16,
         dropout: float = 0.25):
    super(EEGNetLSTM, self).__init__()
    self.name = "EEGNetLSTM"
    self.F1 = F1
    self.F2 = F2
    self.D = D
    self.chunk_size = chunk_size
    self.num_classes = num_classes
    self.num_electrodes = num_electrodes
    self.kernel_1 = kernel_1
    self.kernel_2 = kernel_2
    self.dropout = dropout

    self.block1 = nn.Sequential(
      nn.Conv2d(1, self.F1, (1, self.kernel_1), stride=1, padding=(0, self.kernel_1 // 2), bias=False),
      nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
      Conv2dWithConstraint(self.F1,
                 self.F1 * self.D, (self.num_electrodes, 1),
                 max_norm=1,
                 stride=1,
                 padding=(0, 0),
                 groups=self.F1,
                 bias=False), nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
      nn.ELU(), nn.AvgPool2d((1, 4), stride=4), nn.Dropout(p=dropout))

    self.block2 = nn.Sequential(
      nn.Conv2d(self.F1 * self.D,
            self.F1 * self.D, (1, self.kernel_2),
            stride=1,
            padding=(0, self.kernel_2 // 2),
            bias=False,
            groups=self.F1 * self.D),
      nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
      nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3), nn.ELU(), nn.AvgPool2d((1, 8), stride=8),
      nn.Dropout(p=dropout))

    self.lin = nn.Sequential(
      nn.Linear(self.feature_dim(), 10*num_classes,bias=False),
      nn.ReLU())
    self.lstm = nn.LSTM(input_size=1, hidden_size=10, num_layers=1, dropout=0.4, batch_first=True)
    # Output layer with Softmax activation
    self.output_layer = nn.Sequential(
        nn.Linear(10, num_classes),
        nn.Softmax(dim=1)
    )

  def feature_dim(self):
    with torch.no_grad():
      mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)

      mock_eeg = self.block1(mock_eeg)
      mock_eeg = self.block2(mock_eeg)

    return self.F2 * mock_eeg.shape[3]

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
      x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 60, 151]`. Here, :obj:`n` corresponds to the batch size, :obj:`60` corresponds to :obj:`num_electrodes`, and :obj:`151` corresponds to :obj:`chunk_size`.

    Returns:
      torch.Tensor[number of sample, number of classes]: the predicted probability that the samples belong to the classes.
    """
    x = self.block1(x)
    x = self.block2(x)
    x = x.flatten(start_dim=1)
    x = self.lin(x)
    x = x.view(-1, 40, 1)
    x,_ = self.lstm(x)
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

def test_valid_data_prep(X):
    
    total_X = None
    
    
    # Trimming the data (sample,22,1000) -> (sample,22,800)
    X = X[:,:,0:800]
    
    # Maxpooling the data (sample,22,800) -> (sample,22,800/sub_sample)
    X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, 2), axis=3)
    
    
    total_X = X_max
    
    return total_X

def DatasetLoaders(data_dir='./project_data/project',batch_size=256,augment=False,data_leak=False):
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
        X_train_valid_prep,y_train_valid_prep = train_data_prep(X_train_valid,y_train_valid,2,2,True)
        
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
        x_train, y_train = train_data_prep(x_train_prep, y_train_prep,2,2,True)
        
        ## Preprocessing the other Subsets
        x_valid = test_valid_data_prep(x_valid)
      X_test_prep = test_valid_data_prep(X_test)  
    else:
      ## Simple Truncation of Time-Series
      X_train_valid_prep = X_train_valid[:,:,0:500]
      X_test_prep = X_test[:,:,0:500]
      
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

    # Swapping Axis to Conform with EEGNet
    x_train = np.swapaxes(x_train, 1,3)
    x_train = np.swapaxes(x_train, 3,2)
    x_valid = np.swapaxes(x_valid, 1,3)
    x_valid = np.swapaxes(x_valid, 3,2)
    x_test = np.swapaxes(x_test, 1,3)
    x_test = np.swapaxes(x_test, 3,2)
    
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