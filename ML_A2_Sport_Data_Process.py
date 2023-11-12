import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.stats import skew, kurtosis
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.svm import SVC

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch import optim
from torch.optim import SGD

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

# Define the name list of motions and sensors.
# There are 3 motions and each motion has dataset from 3 sensors.
motions = ['./Jumping_Jack_x10', './Lunges_x10', './Squat_x10']
sensors = ['/Accelerometer', '/Orientation', '/TotalAcceleration']


cols_selected = ['x_Acc', 'y_Acc', 'z_Acc','roll', 'pitch', 'yaw',
                   'x_Total', 'y_Total', 'z_Total','motion']

# Global variable dataframe_loaded
dataframe_loaded = pd.DataFrame()

# Define the size of window and stepsize. 
# The step size is smaller than window size indicating that windows have overlaps
window_size = 10
step_size = 9

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
        #for df in dataframe_loaded.items():
        #original_nan_count = dataframe_loaded.isna().sum().sum()
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
    
        # Print debugging message
        '''        
        print(f"{df_key}:")
        print(f"Original NaN count: {original_nan_count}")
        print(f"NaN count after forward fill: {nan_count_after_ffill}")
        print(f"NaN filled with forward fill: {original_nan_count - nan_count_after_ffill}")
        print(f"NaN count after mean imputation: {nan_count_after_imputing}")
        print(f"NaN filled with mean imputation: {nan_count_after_ffill - nan_count_after_imputing}")
        '''
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

# Define a DecisionTree
class DecisionTree(nn.Module):
    def __init__(self, max_depth):
        super(DecisionTree, self).__init__()
        self.max_depth = max_depth
        self.tree = None
    # This function is responsible for building the decision tree by recursively
    # dividing the training data into multiple leaf nodes.
    # It's used to create a hierarchy of decision nodes.
    def fit(self, X, y, depth=0):
        # If the tress reaches its maximun depth, then just return the
        # the majority of results. 
        if depth >= self.max_depth:
            #print("depth >= self.max_depth")
            # Return the dictionary variable type, to be compatible with the tree.
            return {"class": torch.bincount(y).argmax()}
        # If there's only one leaf, then just return it.
        if len(torch.unique(y)) == 1:
            #print("len(torch.unique(y)) == 1:")
            return {"class": y[0]}

        num_samples, num_features = X.shape
        best_gini = 1.0
        best_split = None
        left_X, right_X, left_y, right_y = None, None, None, None

        for feature_index in range(num_features):
            feature_values = X[:, feature_index]
            unique_values = torch.unique(feature_values)
            for value in unique_values:
                split_mask = feature_values <= value
                left, right = y[split_mask], y[~split_mask]
                # The Gini impurity is a metric commonly used in decision tree
                # to evaluate how well a split separates the data into two
                # groups (left and right).
                # This equation sums up the impurities of the left and right 
                # child nodes, where the weights are the proportions of samples 
                # in each child node relative to the total number of samples in
                # the current node.
                gini = (len(left) / num_samples) * self.gini_impurity(left) + (len(right) / num_samples) * self.gini_impurity(right)
                if gini < best_gini:
                    best_gini = gini
                    best_split = (feature_index, value)
                    left_X, right_X, left_y, right_y = X[split_mask], X[~split_mask], y[split_mask], y[~split_mask]

        if best_gini == 1.0:
            return {"class": torch.bincount(y).argmax()}
        # Define a tree in a dictionary, and it is the return value.
        # Sometimes it returns a node and sometimes it returns a sub-tree.
        # So it's vital to keep the return value of the same time. The node 
        # needs to be defined as a dictionary.
        self.tree = {
            "feature_index": best_split[0],
            "split_value": best_split[1],
            "left": self.fit(left_X, left_y, depth + 1),
            "right": self.fit(right_X, right_y, depth + 1)
        }
        return self.tree

    # Calculate the gini impurity for a node:
    # Gini(S) = 1 − ∑Pi^2
    # Pi is the probability that category i appears in the training sample set S.
    # Referenced by 
    # Hao, Z. and Gou, G., 2019. Survey of machine learning random forest
    # algorithms. In 3rd International Conference on Computer Engineering,
    # Information Science and Internet Technology (pp. 50-59).
    def gini_impurity(self, y):
        if len(y) == 0:
            return 0
        # Calculate the probability.
        p = torch.bincount(y) / len(y)
        return 1 - torch.sum(p ** 2)

    def predict(self, X):
        if self.tree is None:
            raise Exception("The tree is not trained yet.")
        return torch.tensor([self._predict(x, self.tree) for x in X])
    # This is incursively exploring all branches and all nodes of the tree.
    def _predict(self, x, node):
        if "split_value" not in node:
            # Return the class lable, which is also the node.
            return node["class"]
        if x[node["feature_index"]] <= node["split_value"]:
            return self._predict(x, node["left"])
        else:
            return self._predict(x, node["right"])

# Create a RandomForeset with multiple tress.
class RandomForestClassifier:
    def __init__(self, num_trees, max_depth):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.trees = [DecisionTree(max_depth) for _ in range(num_trees)]
    # In the random forest, it implements all functions by calling the funcs
    # in decision trees.
    def fit(self, X, y):
        for tree in self.trees:
            tree.fit(X, y)
    
    def predict(self, X):
        predictions = [tree.predict(X) for tree in self.trees]
        return torch.stack(predictions, dim=0).mode(0).values

def plot_3d_surface(X, y, z):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Use coolwarm to plot the surface and antialiased=False to disable transparency.
    surf = ax.plot_surface(X, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # set the range for z axis.
    ax.set_zlim(0, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # add a colorbar to indicate the meaning of color.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    
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


def create_train_evaluate_RF(x, y) :
    print(f"create_train_evaluate_RF with numbers_of_tree = {x} and depth = {y}")
    random_forest = RandomForestClassifier(x, y)
    random_forest.fit(X_train, y_train)
    predictions = random_forest.predict(X_test)
    accuracy, precision, recall, f1 =  evaluate_classifer(
        y_test, predictions, False)
    print(f"evaluating results: accuracy({accuracy}),precision({precision}),recall({recall}),f1({f1})")
    return accuracy, precision, recall, f1

# Exhausive tuning for different hyper parameters for Random Forest.
def tune_hyperparameters_for_RF(
        num_trees, max_depth, X_train, X_test, y_train, y_test) :
    best_num_tree = num_trees
    best_max_depth = max_depth
    numbers_of_tree = np.arange(best_num_tree, 10, 2)
    max_depths = np.arange(best_max_depth, 10, 2)
    x, y = np.meshgrid(numbers_of_tree, max_depths)
    ufunc_hyper_tuning = np.frompyfunc(create_train_evaluate_RF, 2, 4)
    accuracy_list, rf_precision_list, rf_recall_list, rf_f1_list = ufunc_hyper_tuning(x, y)
    plot_3d_surface(x, y, accuracy_list)
    max_index = np.unravel_index(np.argmax(rf_f1_list), rf_f1_list.shape)
    best_num_tree, best_max_depth = x[max_index], y[max_index]
    print(f" best_num_tree = {best_num_tree}, best_max_depth = {best_max_depth}")
    return best_num_tree, best_max_depth




class LinearSVM(nn.Module):
    def __init__(self):
        super(LinearSVM, self).__init__()
        self.fc = nn.Linear(54, 1)

    def forward(self, x):
        return self.fc(x)


criterion = nn.HingeEmbeddingLoss()


def evaluate_model(model, X, Y):
    model.eval()

    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)

        # 计算准确率
        total = Y.size(0)
        correct = (predicted == Y).sum().item()
        accuracy = correct / total

    return accuracy


if __name__=="__main__":
    dataframe_loaded = load_data_from_files()
    print(dataframe_loaded)
    dataframe_loaded = data_preprocess(dataframe_loaded)
    dataframe_scaled = feature_scaling(dataframe_loaded)
    dataframe_meaned = mean_filter(dataframe_scaled)
    dataframe_noised = add_noise(dataframe_meaned)
    features_list = feature_extraction(dataframe_noised)
    
    # Convert the list to data frame to save in a csv file.
    feature_df = pd.DataFrame(features_list)
    feature_df.to_csv('feature_df.csv', sep=',', index=False, encoding='utf-8')
    
    # Split X from y to do data spliting.
    X = feature_df.iloc[:, :-1]
    y = feature_df.iloc[:, -1]

    X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X, y, test_size=0.4, random_state=42)


    # Create an SVM classifier instance based on sklearn
    # Set the parameter grid that you want to tune
    param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    # Create an SVM classifier instance
    svm_classifier = SVC()
    # Create a GridSearchCV instance
    grid_search = GridSearchCV(svm_classifier, param_grid, cv=5)  # 5 fold cross verification
    # Perform grid search and cross-validation
    grid_search.fit(X_train_data, y_train_data)
    # Print optimum parameter
    print("SVM Best parameters:", grid_search.best_params_)
    # Make predictions on the test set using the best parameters
    y_pred = grid_search.predict(X_test_data)
    # Generate and print detailed classification reports
    evaluate_classifer(y_test_data, y_pred, True, "SVM")

    # Convert data to tensor type.
    X_train = torch.tensor(X_train_data.to_numpy(), dtype=torch.float32)
    X_test = torch.tensor(X_test_data.to_numpy(), dtype=torch.float32)
    y_train = torch.tensor(y_train_data.to_numpy(), dtype=torch.long)
    y_test = torch.tensor(y_test_data.to_numpy(), dtype=torch.long)



    #-- torch SVM-----
    # Train an SVM for each class
    models = [LinearSVM() for _ in range(3)]
    optimizers = [optim.SGD(model.parameters(), lr=0.1) for model in models]
    for i, model in enumerate(models):
        for epoch in range(1000):
            optimizers[i].zero_grad()
            output = model(X_train).squeeze()
            target = torch.where(y_train == i, 1, -1).float()  # The current category is 1 and others are -1
            loss = criterion(output, target)
            loss.backward()
            optimizers[i].step()

        y_pre = model(X_test)
        accuracy = evaluate_model(model, X_test, y_test)
        print(f'Accuracy: {accuracy * 100:.2f}%')



    #Create and train a Random Forest.
    #Make an initiall guess for number of trees and maximum depth.
    num_trees = 5
    max_depth = 30
    best_num_tree = num_trees
    best_max_depth = max_depth
    #best_num_tree, best_max_depth = tune_hyperparameters_for_RF(
    #    num_trees, max_depth, X_train, X_test, y_train, y_test)

    print("1 Create RandomForestClassifer===========")

    random_forest = RandomForestClassifier(best_num_tree, best_max_depth)
    print("1 Fit RandomForestClassifer===========")
    random_forest.fit(X_train, y_train)
    print("1 RandomForestClassifer predicts()===========")
    rf_predictions = random_forest.predict(X_test)
    evaluate_classifer(y_test, rf_predictions, True, "random_forest")
    print(y_train)
    print("===========")
    print(y_test)
    print("===========")
    print(rf_predictions)

    random_forest = RandomForestClassifier(best_num_tree * 2, best_max_depth)
    print("2 Fit RandomForestClassifer===========")
    random_forest.fit(X_train, y_train)
    print("2 RandomForestClassifer predicts()===========")
    rf_predictions = random_forest.predict(X_test)
    evaluate_classifer(y_test, rf_predictions, True, "random_forest")
    print(y_train)
    print("===========")
    print(y_test)
    print("===========")
    print(rf_predictions)

    random_forest = RandomForestClassifier(best_num_tree * 4, best_max_depth)
    print("3 Fit RandomForestClassifer===========")
    random_forest.fit(X_train, y_train)
    print("3 RandomForestClassifer predicts()===========")
    rf_predictions = random_forest.predict(X_test)
    evaluate_classifer(y_test, rf_predictions, True, "random_forest")
    print(y_train)
    print("===========")
    print(y_test)
    print("===========")
    print(rf_predictions)
