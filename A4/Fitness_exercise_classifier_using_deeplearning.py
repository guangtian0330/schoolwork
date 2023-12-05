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
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit

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

"""

class CNNClassifier(nn.Module):
    def __init__(self, conv_num=2, kernel_size=3):
        super(CNNClassifier, self).__init__()
        self.conv_num = conv_num
        if conv_num == 2:
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=kernel_size, padding=1)
            self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=kernel_size, padding=1)
            # pooling layer
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(in_features=416, out_features=10)
            self.fc2 = nn.Linear(in_features=10, out_features=3)
        elif conv_num == 3:
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=kernel_size, padding=1)
            self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=kernel_size, padding=1)
            self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=kernel_size, padding=1)
            # pooling layer
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(in_features=192, out_features=10)
            self.fc2 = nn.Linear(in_features=10, out_features=3)
        elif conv_num == 4:
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=kernel_size, padding=1)
            self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=kernel_size, padding=1)
            self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=kernel_size, padding=1)
            self.conv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=kernel_size, padding=1)
            # pooling layer
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(in_features=96, out_features=10)
            self.fc2 = nn.Linear(in_features=10, out_features=3)

    def forward(self, x):
        if self.conv_num == 2:
            # Apply convolution layer and ReLU activation function
            x = torch.relu(self.conv1(x))
            # pooling layer
            x = self.pool(x)
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            # Flattening features to input into the fully connected layer
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            # Application output layer
            x = self.fc2(x)
        elif self.conv_num == 3:
            # Apply convolution layer and ReLU activation function
            x = torch.relu(self.conv1(x))
            # pooling layer
            x = self.pool(x)
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = torch.relu(self.conv3(x))
            x = self.pool(x)
            # Flattening features to input into the fully connected layer
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            # Application output layer
            x = self.fc2(x)
        elif self.conv_num == 4:
            # Apply convolution layer and ReLU activation function
            x = torch.relu(self.conv1(x))
            # pooling layer
            x = self.pool(x)
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = torch.relu(self.conv3(x))
            x = self.pool(x)
            x = torch.relu(self.conv4(x))
            x = self.pool(x)
            # Flattening features to input into the fully connected layer
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            # Application output layer
            x = self.fc2(x)
        return x

def train_cnn_model(model, input_seq, target_seq, criterion, optimizer, epochs=500) :
    input_seq = input_seq.unsqueeze(1)
    model.train()
    loss_result = []
    batch_size = 10
    for epoch in range(100):
        total_loss = 0.0
        for i in range(0, len(X_train), batch_size):
            # Get batch data
            inputs = input_seq[i:i + batch_size]
            labels = target_seq[i:i + batch_size]

            # Clear the old gradient
            optimizer.zero_grad()

            # Forward propagation
            outputs = model(inputs)

            # Calculate the loss
            loss = criterion(outputs, labels)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss.item()

        loss_result.append(total_loss)
        print(f'Epoch {epoch + 1}, Loss: {total_loss:.4f}')
    return loss_result
# Predicting
def cnn_predict(model, inputs):
    predicts = []
    inputs = inputs.unsqueeze(1)
    with torch.no_grad():
        for i in range(0, len(inputs)):
            outputs = model(inputs[i])
            _, predicted_label = torch.max(outputs, 1)
            predicts.append(predicted_label.item())
    return torch.tensor(predicts, dtype=torch.long)


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


def hyperparam_tuning(param_grid, X_train, y_train, scoring='accuracy'):

  # Create a classifier model which can be accepted by sklearner.
  gru_model = NeuralNetClassifier(
      GRUClassifier,
      # Use CrossEntropyLoss as the loss func
      criterion=nn.CrossEntropyLoss,
      optimizer=optim.Adam,
      max_epochs=100,
      batch_size=32,
      # The last element is the input size.
      module__input_size=X_train.shape[2],
      module__output_size=3
  )

  # Cross validation for time series data with 5 folds.
  tscv = TimeSeriesSplit(n_splits=5)

  # Use GridSearchCV to search the optimal hyperparameter
  grid = GridSearchCV(estimator=gru_model, param_grid=param_grid, cv=tscv, scoring=scoring, verbose=1)
  grid_result = grid.fit(X_train, y_train)

  # Print the beat parameter
  print("Best Parameters: ", grid_result.best_params_)
  return grid_result.best_params_


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

def evaluate_with_parameters(param_grid) :
    best_parameters = hyperparam_tuning(param_grid, X_train.unsqueeze(1), y_train, scoring='f1_weighted')
    print("best_parameters = ",best_parameters)
    dropout_rate = best_parameters['module__dropout_rate']
    hidden_size = best_parameters['module__hidden_size']
    num_layers = best_parameters['module__num_layers']
    gru_model = GRUClassifier(input_size, hidden_size, output_size)
    gru_optimizer = torch.optim.Adam(gru_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    loss_list = train_model(gru_model, X_train, y_train, criterion, gru_optimizer)

    # Test the GRU Model
    gru_test_outputs = predict(gru_model,X_test.unsqueeze(1))

    evaluate_classifer(y_test, gru_test_outputs, True, "GRU")
    print(loss_list)
    plt.figure(figsize=(12, 6))
    plt.plot(range(0, 1000, 1), loss_list, label="GRU Predicted Loss func")  # Add this line for GRU
    plt.legend()
    plt.title(f"GRU Loss with 'module__dropout_rate': {dropout_rate} 'module__hidden_size': {hidden_size}, 'module__num_layers': {num_layers}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Function Values")
    plt.grid(True)
    plt.show()
    return dropout_rate, hidden_size, num_layers



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
    cnn_model = CNNClassifier(2)
    cnn_optimizer = optim.SGD(cnn_model.parameters(), lr=0.01)
    cnn_criterion = nn.CrossEntropyLoss()
    loss_result = train_cnn_model(cnn_model, X_train, y_train, cnn_criterion, cnn_optimizer)
    cnn_predicts = cnn_predict(cnn_model,X_test.unsqueeze(1))
    evaluate_classifer(y_test, cnn_predicts, True, "CNN")

    plt.figure(figsize=(12, 6))
    plt.plot(range(0, 100, 1), loss_result, label="CNN Predicted Loss func")
    plt.legend()
    plt.title(f"CNN Loss with 'conv': {2} 'kernel size': {3}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Function Values")
    plt.grid(True)
    plt.show()

    cnn_model = CNNClassifier(3)
    cnn_optimizer = optim.SGD(cnn_model.parameters(), lr=0.01)
    cnn_criterion = nn.CrossEntropyLoss()
    loss_result = train_cnn_model(cnn_model, X_train, y_train, cnn_criterion, cnn_optimizer)
    cnn_predicts = cnn_predict(cnn_model, X_test.unsqueeze(1))
    evaluate_classifer(y_test, cnn_predicts, True, "CNN")

    plt.figure(figsize=(12, 6))
    plt.plot(range(0, 100, 1), loss_result, label="CNN Predicted Loss func")
    plt.legend()
    plt.title(f"CNN Loss with 'conv': {3} 'kernel size': {3}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Function Values")
    plt.grid(True)
    plt.show()

    cnn_model = CNNClassifier(4)
    cnn_optimizer = optim.SGD(cnn_model.parameters(), lr=0.01)
    cnn_criterion = nn.CrossEntropyLoss()
    loss_result = train_cnn_model(cnn_model, X_train, y_train, cnn_criterion, cnn_optimizer)
    cnn_predicts = cnn_predict(cnn_model, X_test.unsqueeze(1))
    evaluate_classifer(y_test, cnn_predicts, True, "CNN")

    plt.figure(figsize=(12, 6))
    plt.plot(range(0, 100, 1), loss_result, label="CNN Predicted Loss func")
    plt.legend()
    plt.title(f"CNN Loss with 'conv': {3} 'kernel size': {3}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Function Values")
    plt.grid(True)
    plt.show()
    #---------------------------------------------------------------------------
    
    output_size = 3
    input_size = 54
    learning_rate = 0.01
    
    # Define the hyperparameter space.
    param_grid = {
        'module__dropout_rate': [0.2, 0.5],
        # With a dropour rate, there are at least 2 layers.
        'module__num_layers': [2, 3],
        'module__hidden_size': [50, 100],
    }
    # Tune ther parameters and evaluate the model.
    evaluate_with_parameters(param_grid)

    
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
