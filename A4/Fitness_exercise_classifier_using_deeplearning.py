# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 17:04:46 2023

@author: Guangtian Gong, Sisong Li, Zijian Han
"""
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.stats import skew, kurtosis
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Define the name list of motions and sensors.
# There are 3 motions and each motion has dataset from 3 sensors.
motions = ['./JJx10-2023-11-19_19-24-30',
           './Lux10-2023-11-19_19-33-46',
           './Sqx10-2023-11-19_19-26-29']
sensors = ['/Accelerometer', '/Orientation', '/TotalAcceleration']

cols_selected = ['x_Acc', 'y_Acc', 'z_Acc','roll', 'pitch', 'yaw',
                   'x_Total', 'y_Total', 'z_Total','motion']

# Global variable dataframe_loaded
dataframe_loaded = pd.DataFrame()

# Define the size of window and stepsize. 
# The step size is smaller than window size indicating that windows have overlaps
window_size = 5
step_size = 3

'''
load files for a referred motion and return a dataframe.
'time' and 'seconds_elapsed' are removed after data is loaded.
They are dropped because:
It doesn't involve any predictive information.
'''
def loadFileForMotion(motion) :
    filenames = [f"{motion}{sensor}.csv" for sensor in sensors]
    #print(filenames)
    dataframes_read = [pd.read_csv(filename).drop(["seconds_elapsed", "time"], axis=1) for filename in filenames]
    #dataframes_read[2].head(5)
    dataframe = dataframes_read[0].join(dataframes_read[1],lsuffix='_Acc', rsuffix='_Total')
    dataframe = dataframe.join(dataframes_read[2],lsuffix='_Acc', rsuffix='_Total')
    dataframe['motion'] = motion
    return dataframe

'''
 The returned framedata combines all features from all sensors for each motion
 with the axis=1 and then appends all motions in the list.
'''
def load_data_from_files() :
    dataframe_loaded = pd.concat(
        [loadFileForMotion(motions[0]),
         loadFileForMotion(motions[1]),
         loadFileForMotion(motions[2])])
    dataframe_loaded = dataframe_loaded[cols_selected]
    return dataframe_loaded

'''
 Data preprocess.  This function drop duplicates, fulfill NaN with mean values.
'''
def data_preprocess(dataframe_loaded):
    print("==========ENTER  data_process <<<<<<<<<<<")
    if not dataframe_loaded.empty:
        # For df in dataframe_loaded.items():
        # original_nan_count = dataframe_loaded.isna().sum().sum()
        dataframe_loaded.drop_duplicates(inplace=True)
        dataframe_loaded.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataframe_loaded.ffill(inplace=True)
            
        # split 'motion' column before fullfill NaN with mean values, 
        # since 'motion' column is not numerical.
        y = dataframe_loaded['motion']
        X = dataframe_loaded.drop('motion', axis=1)
        
        # Create an Inputer and fulfuill NaN with mean values.
        # Apply this imputer to all columns.
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
        # Before combining X and y it's suggested to reset index, otherwise
        # they will have common index.
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
        print(">>>>>>>>>>>LEAVE  data_process ==============")
        return pd.concat([X, y], axis=1)
    else:
        print(">>>>>>>>>>>LEAVE  data_process ==============")
        return dataframe_loaded

def add_noise(dataframe_loaded) :
    noise_level = 0.1
    for col in dataframe_loaded.columns:
        if col != 'motion':
            dataframe_loaded[col] = dataframe_loaded[col] + np.random.normal(0, noise_level, size=dataframe_loaded[col].shape)
    return dataframe_loaded

def mean_filter(dataframe_loaded):
    print("==========ENTER  mean_filter <<<<<<<<<<<")
    # Mean filtering
    for col in cols_selected:
        dataframe_loaded[col] = dataframe_loaded[col].rolling(window=window_size, min_periods=1).mean()
    print(">>>>>>>>>>>LEAVE  mean_filter ==============")
    return dataframe_loaded
       
# We choose to extract features 'x_Acc', 'y_Acc', 'z_Acc','roll', 'pitch', 'yaw',
# 'x_Total', 'y_Total', 'z_Total', where x_Acc is from the x value for accelerometer,
# and x_Total is from the total accelerate. We change the name to discriminate them.
# euler angles('roll', 'pitch', 'yaw') contain the same information with quaternion values so
# we only choose the euler angles because they are more intuitive and easier to
# understand.
def feature_extraction(dataframe_loaded):
    print("==========ENTER  feature_extraction <<<<<<<<<<<")    
    features = []
    cols = ['x_Acc', 'y_Acc', 'z_Acc','roll', 'pitch', 'yaw',
            'x_Total', 'y_Total', 'z_Total']
    for start in range(0, dataframe_loaded.shape[0] - window_size + 1, step_size):
        end = start + window_size
        window_df = dataframe_loaded.iloc[start:end]
        line = [value 
                for col in cols
                for value in [window_df[col].mean(),
                          window_df[col].std(),
                          skew(window_df[col], nan_policy='omit'),
                          kurtosis(window_df[col], nan_policy='omit'),
                          window_df[col].max(),
                          window_df[col].min()]
        ]
        line.append(round(window_df['motion'].iloc[0]))
        features.append(line)
    print(">>>>>>>>>>>LEAVE  feature_extraction ==============")
    return features


# Feature scaling is necessary to SVM which is more sensitve to feature scalings
# but for Random Forest this scaling operation is unecessary.
def feature_scaling(feature_df) :
    print("==========ENTER  feature_scaling <<<<<<<<<<<")    
    # Fulfill NaN values with mean values.
    imputer = SimpleImputer(strategy='mean')

    # Split motion and other columns so we can scale the data parts.
    y = dataframe_loaded['motion']
    X = dataframe_loaded.drop('motion', axis=1)
    
    # Feature scaling
    X_imputed = imputer.fit_transform(X)
    # SVM is more sensitive to feature scales so the standardScaler can a better fit.
    scaler = StandardScaler()
    #scaler = MinMaxScaler()
    X_imputed_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

    # Use label encoder to transform motion column to numerical values.
    label_encoder = LabelEncoder()
    y = pd.DataFrame(label_encoder.fit_transform(y), columns = ['motion'])
 
    # Combine X and y before returnning it.
    X_imputed_scaled.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    feature_numeric_scaled = pd.concat([X_imputed_scaled, y], axis=1)
    print(">>>>>>>>>>>LEAVE  feature_scaling ==============")
    return feature_numeric_scaled


"""
Develop and train 1D CNN (Convolutional Neural Network) models appropriate for
sequential sensor data classification.

TBD by Sisong
"""

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Define the architecture of the neural network
        # Convolutional Layer 1: Input channels=3 (RGB), Output channels=32, Kernel size=3x3, Padding=1
        # The neural network learns these weights through backpropagation during training
        self.conv1 = nn.Conv1d(54, 32, 3, padding=1)
        # Convolutional Layer 2: Input channels=32, Output channels=64, Kernel size=3x3, Padding=1
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        # Max Pooling Layer: Kernel size=2x2, Stride=2 (pooling window moves horizontally and vertically after each operation)
        self.pool = nn.MaxPool2d(2, 2) # now the image is 8x8
        # Fully Connected Layer 1: Input features=64*8*8 (output channels * output image size), Output features=128
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        # Fully Connected Layer 2 (Output Layer): Input features=128, Output features=10 (number of classes)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Forward pass through the network
        # Convolutional Layer 1 followed by ReLU activation and max pooling
        x= self.conv1(x)
        x = self.pool(torch.relu(x))
        # Convolutional Layer 2 followed by ReLU activation and max pooling
        x = self.pool(torch.relu(self.conv2(x)))
        # Reshape the tensor for the fully connected layers
        x = x.view(-1, 64 * 8 * 8)
        # Fully Connected Layer 1 followed by ReLU activation
        x = torch.relu(self.fc1(x))
        # Fully Connected Layer 2 (Output Layer)
        x = self.fc2(x)

        return x

def train_cnn_model(model, input_seq, target_seq, criterion, optimizer, epochs=500) :
  # input_seq = input_seq.unsqueeze(1)
  input_seq = torch.transpose(input_seq, 0, 1)

  print(f"input_seq shape = {input_seq.shape}")
  for epoch in range(epochs):
      # Reset the gradient to zero in case of accumulation.
      optimizer.zero_grad()
      # Forward propogation
      output = model(input_seq)
      # Calculate the loss using criterion.
      loss = criterion(output, target_seq)
      # Calculate the backpropogation
      loss.backward()
      # Update the weightes based on the gradients.
      optimizer.step()
      # Print the debugging information for loss function
      if epoch % 20 == 0:
          print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
# Predicting
def cnn_predict(model, inputs):
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted_labels = torch.max(outputs, 1)
    return predicted_labels


"""
Develop and train GRU (Gated Recurrent Unit) networks for sequential sensor
data classification.
"""
class GRUClassifier(nn.Module) :
    # number of layers is 1 by default, and dropout_rate is 0 by default.
    def __init__(self, input_size, hidden_size, output_size,
                 num_layers=1, dropout_rate=0) :
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers=num_layers,
            dropout=dropout_rate, batch_first = True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x) :
        gru_out, _ = self.gru(x)
        print(gru_out.shape)
        last_output = gru_out[:,-1, :]
        output = self.fc(last_output)
        output_probs = self.softmax(output)
        return output_probs


def train_model(model, input_seq, target_seq, criterion, optimizer, epochs=500) :
  input_seq = input_seq.unsqueeze(1)
  print(f"input_seq shape = {input_seq.shape}")
  for epoch in range(epochs):
      # Reset the gradient to zero in case of accumulation.
      optimizer.zero_grad()
      # Forward propogation
      output = model(input_seq)
      # Calculate the loss using criterion.
      loss = criterion(output, target_seq)
      # Calculate the backpropogation
      loss.backward()
      # Update the weightes based on the gradients.
      optimizer.step()
      # Print the debugging information for loss function
      if epoch % 20 == 0:
          print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
# Predicting
def predict(model, inputs):
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted_labels = torch.max(outputs, 1)
    return predicted_labels


def evaluate_classifer(y_test, y_predictions, if_print_detailed_report, algorithm) :
    # Calculate the accuracy.
    accuracy = accuracy_score(y_test, y_predictions)
    # Calculate the precision, recall rate and f1 score.
    rf_precision = precision_score(y_test, y_predictions, average='weighted')
    rf_recall = recall_score(y_test, y_predictions, average='weighted')
    rf_f1 = f1_score(y_test, y_predictions, average='weighted')
    
    if if_print_detailed_report :
        # Generate the confusion matrix.
        rf_confusion_matrix = confusion_matrix(y_test, y_predictions)
        # Print the report if necessary.
        print(f"{algorithm}Accuracy:", accuracy)
        print(f"{algorithm} Confusion Matrix:\n{rf_confusion_matrix}")
        print(f"{algorithm} Precision: {rf_precision}")
        print(f"{algorithm} Recall: {rf_recall}")
        print(f"{algorithm} F1 Score: {rf_f1}")
        # Print the classifer analysis if necessary.
        print(f"{algorithm} Classifier Report:")
        print(classification_report(y_test, y_predictions))
    return accuracy, rf_precision, rf_recall, rf_f1

if __name__ == '__main__':
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataframe_loaded = load_data_from_files()
    print(dataframe_loaded)
    dataframe_loaded = data_preprocess(dataframe_loaded)
    dataframe_scaled = feature_scaling(dataframe_loaded)
    dataframe_meaned = mean_filter(dataframe_scaled)
    dataframe_noised = add_noise(dataframe_meaned)
    
    # Use previous solution to extract the numerical characteristics.
    features_list = feature_extraction(dataframe_noised)
    # Convert the list to data frame to save in a csv file.
    feature_df = pd.DataFrame(features_list)
    feature_df.to_csv('feature_df.csv', sep=',', index=False, encoding='utf-8')
    #feature_df = scaler.fit_transform(feature_df)
    # Split X from y to do data spliting.
    X = feature_df.iloc[:, :-1]
    y = feature_df.iloc[:, -1]

    X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X, y, test_size=0.4, random_state=42)
    # Convert data to tensor type.
    X_train = torch.tensor(X_train_data.to_numpy(), dtype=torch.float32)
    X_test = torch.tensor(X_test_data.to_numpy(), dtype=torch.float32)
    y_train = torch.tensor(y_train_data.to_numpy(), dtype=torch.long)
    y_test = torch.tensor(y_test_data.to_numpy(), dtype=torch.long)

    #-----------CNN-------------------

    net = CNN()

    # Define the loss function (Cross Entropy Loss for classification problems)
    criterion = nn.CrossEntropyLoss()
    # Define the optimizer (Stochastic Gradient Descent with momentum)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    # Load and preprocess the CIFAR-10 dataset
    # a normalized PyTorch tensor that can be passed as input
    # torchvision.transforms.Normalize(mean, std, inplace=False)
    # Suppose you have a pixel value of (100, 150, 200) for a particular pixel in an image.
    # After applying this normalization, the pixel values would be transformed as follows:
    # Red channel: (100 - 0.5) / 0.5 = 199
    # Green channel: (150 - 0.5) / 0.5 = 299
    # Blue channel: (200 - 0.5) / 0.5 = 399
    # train_cnn_model(net, X_train, y_train, criterion, optimizer)


    #---------------------------------------------------------------------------
    
    output_size = 3
    hidden_size = 64
    input_size = 54
    learning_rate = 0.01
    gru_model = GRUClassifier(input_size, hidden_size, output_size)
    gru_optimizer = torch.optim.Adam(gru_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    print("---------------------------")
    print(X_train.shape)
    train_model(gru_model, X_train, y_train, criterion, gru_optimizer)
    
    # Test the GRU Model
    #gru_model.eval()
    gru_test_outputs = predict(gru_model,X_test.unsqueeze(1))
    #gru_test_outputs = gru_test_outputs.detach().numpy()
    print(gru_test_outputs)
    evaluate_classifer(y_test, gru_test_outputs, True, "GRU")