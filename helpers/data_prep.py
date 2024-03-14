import torch
import numpy as np
from keras.utils import to_categorical

def train_data_prep(X,y,sub_sample,average,noise,chunk_size=800,verbose=False):
    
    total_X = None
    total_y = None
    
    # Trimming the data (sample,22,1000) -> (sample,22,800)
    X = X[:,:,0:chunk_size]
    if(verbose):
      print('Shape of X after trimming:',X.shape)
    
    # Maxpooling the data (sample,22,800) -> (sample,22,800/sub_sample)
    X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, sub_sample), axis=3)
    
    
    total_X = X_max
    total_y = y
    if(verbose):
      print('Shape of X after maxpooling:',total_X.shape)
    
    # Averaging + noise 
    X_average = np.mean(X.reshape(X.shape[0], X.shape[1], -1, average),axis=3)
    X_average = X_average + np.random.normal(0.0, 0.5, X_average.shape)
    
    total_X = np.vstack((total_X, X_average))
    total_y = np.hstack((total_y, y))
    if(verbose):
      print('Shape of X after averaging+noise and concatenating:',total_X.shape)
    
    # Subsampling
    
    for i in range(sub_sample):
        
        X_subsample = X[:, :, i::sub_sample] + \
                            (np.random.normal(0.0, 0.5, X[:, :,i::sub_sample].shape) if noise else 0.0)
            
        total_X = np.vstack((total_X, X_subsample))
        total_y = np.hstack((total_y, y))
        
    if(verbose):
      print('Shape of X after subsampling and concatenating:',total_X.shape)
      print('Shape of Y:',total_y.shape)
    return total_X,total_y

def test_valid_data_prep(X, chunk_size=800,verbose=False):
    
    total_X = None
    
    
    # Trimming the data (sample,22,1000) -> (sample,22,800)
    X = X[:,:,0:chunk_size]
    if(verbose):
      print('Shape of X after trimming:',X.shape)
    
    # Maxpooling the data (sample,22,800) -> (sample,22,800/sub_sample)
    X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, 2), axis=3)
    
    
    total_X = X_max
    if(verbose):
      print('Shape of X after maxpooling:',total_X.shape)
    
    return total_X
    
def DatasetLoaders(data_dir='./project_data/project',batch_size=256,augment=False,chunk_size=800,add_width=True,eegnet=False):
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
    
    ## Random splitting and reshaping the data
    # First generating the training and validation indices using random splitting
    ind_valid = np.random.choice(2115, 500, replace=False)
    ind_train = np.array(list(set(range(2115)).difference(set(ind_valid))))
    
    # Creating the training and validation sets using the generated indices
    (x_train, x_valid) = X_train_valid[ind_train], X_train_valid[ind_valid] 
    (y_train, y_valid) = y_train_valid[ind_train], y_train_valid[ind_valid]

    x_valid = test_valid_data_prep(x_valid, chunk_size=chunk_size)
    X_test = test_valid_data_prep(X_test, chunk_size=chunk_size)
    if(augment): # Apply Augmentation to Training Set Only
        x_train, y_train = train_data_prep(x_train, y_train,2,2,True, chunk_size=chunk_size)
    else:
        x_train = test_valid_data_prep(x_train, chunk_size=chunk_size)
  
    # Converting the labels to categorical variables for multiclass classification
    y_train = to_categorical(y_train, 4)
    y_valid = to_categorical(y_valid, 4)
    y_test = to_categorical(y_test, 4)
    
    # Adding width of the segment to be 1
    if(add_width):
      x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
      x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], x_train.shape[2], 1)
      x_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    else:
      x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2])
      x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], x_train.shape[2])
      x_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

    # Swapping Axis to Conform with EEGNet
    if(eegnet):
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

    
def SubjectLoaders(data_dir='./project_data/project',batch_size=256,augment=False,chunk_size=800,add_width=True,eegnet=False,subject=1):
    """ Function to Load in the Datasets for Preprocessing for a Specific Subject """
    ## Loading the dataset
    X_test = np.load(f"{data_dir}/X_test.npy")
    y_test = np.load(f"{data_dir}/y_test.npy")
    person_train_valid = np.load(f"{data_dir}/person_train_valid.npy")
    X_train_valid = np.load(f"{data_dir}/X_train_valid.npy")
    y_train_valid = np.load(f"{data_dir}/y_train_valid.npy")
    person_test = np.load(f"{data_dir}/person_test.npy")

    # Filter with Only Particular Subject
    idx_train_valid = np.where(person_train_valid  == subject)[0]
    X_train_valid = X_train_valid[idx_train_valid]
    y_train_valid = y_train_valid[idx_train_valid]
    idx_test = np.where(person_test == subject)[0]
    X_test = X_test[idx_test]
    y_test = y_test[idx_test]
    
    ## Adjusting the labels so that 
    
    # Cue onset left - 0
    # Cue onset right - 1
    # Cue onset foot - 2
    # Cue onset tongue - 3
    
    y_train_valid -= 769
    y_test -= 769
    
    ## Random splitting and reshaping the data
    # First generating the training and validation indices using random splitting
    ind_valid = np.random.choice(X_train_valid.shape[0], int(0.1*X_train_valid.shape[0]), replace=False)
    ind_train = np.array(list(set(range(X_train_valid.shape[0])).difference(set(ind_valid))))
    
    # Creating the training and validation sets using the generated indices
    (x_train, x_valid) = X_train_valid[ind_train], X_train_valid[ind_valid] 
    (y_train, y_valid) = y_train_valid[ind_train], y_train_valid[ind_valid]

    x_valid = test_valid_data_prep(x_valid, chunk_size=chunk_size)
    X_test = test_valid_data_prep(X_test, chunk_size=chunk_size)
    if(augment): # Apply Augmentation to Training Set Only
        x_train, y_train = train_data_prep(x_train, y_train,2,2,True, chunk_size=chunk_size)
    else:
        x_train = test_valid_data_prep(x_train, chunk_size=chunk_size)
  
    # Converting the labels to categorical variables for multiclass classification
    y_train = to_categorical(y_train, 4)
    y_valid = to_categorical(y_valid, 4)
    y_test = to_categorical(y_test, 4)
    
    # Adding width of the segment to be 1
    if(add_width):
      x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
      x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], x_train.shape[2], 1)
      x_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    else:
      x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2])
      x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], x_train.shape[2])
      x_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

    # Swapping Axis to Conform with EEGNet
    if(eegnet):
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