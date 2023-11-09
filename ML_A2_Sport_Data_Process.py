import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.stats import skew, kurtosis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.utils.validation import column_or_1d
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
window_size = 100
step_size = 50

'''
load files for a referred motion and return a dataframe.
'time' and 'seconds_elapsed' are removed after data is loaded.
They are dropped because:
It doesn't involve any predictive information.
不携带预测性信息：如果时间戳本身与目标变量（在您的案例中是运动类型）无关，则它们不会为模型提供任何有价值的信息。比如，如果运动类型不随时间变化而变化，那么时间特征可能对模型预测不会有帮助。
数据泄露：如果时间相关的特征与数据收集的方式有关，而不是与预测目标本身有关，那么它们可能会导致数据泄露，使模型过度拟合训练集中的“时间”模式，而这些模式在未来的数据中可能并不适用。
模型复杂性：保留不相关或不重要的特征会增加模型的复杂性，可能导致模型训练时间更长，而且如果特征过多还可能导致维度灾难。
时间序列问题：如果数据是时间序列数据，'time' 和 'seconds_elapsed' 可能需要以不同的方式处理，如通过时间序列分析而不是标准的机器学习模型。对于这种数据，通常会使用特殊的时间序列模型或将时间信息转换为其他更有用的特征。
在一些情况下，时间特征可以经过适当的工程转换为模型可用的形式。例如，如果您知道某些运动类型在一天中的特定时间更有可能发生，您可以将时间戳转换为一天中的小时来捕获这种周期性。或者，如果您有理由相信运动类型会随着时间推移而发生变化（例如，在长时间的数据收集期间），那么时间也可能成为一个重要特征。
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

def add_noise() :
    noise_level = 0.01  # 噪声水平可以根据需要调整
    for df_key, df in dataframe_loaded.items():
        for col in df.columns:
            if col not in ['motion', 'sensor']:  # 仅对数值特征添加噪声
                df[col] = df[col] + np.random.normal(0, noise_level, size=df[col].shape)

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
            # Return the dictionary variable type, to be compatible with the tree.
            return {"class": torch.bincount(y).argmax()}
        # If there's only one leaf, then just return it.
        if len(torch.unique(y)) == 1:
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

    '''
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.sin(R)
    '''
    # 绘制曲面图
    # 绘制使用冷暖色图着色的 3D 表面。通过使用 antialiased=False 使表面变得不透明。
    surf = ax.plot_surface(X, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # 定制z轴
    ax.set_zlim(0, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # 添加一个颜色条形图展示颜色区间
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    
def evaluate_classifer(y_test, y_predictions, if_print_detailed_report) :
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
        print("Accuracy:", accuracy)
        print(f"Random Forest Confusion Matrix:\n{rf_confusion_matrix}")
        print(f"Random Forest Precision: {rf_precision}")
        print(f"Random Forest Recall: {rf_recall}")
        print(f"Random Forest F1 Score: {rf_f1}")
        # Print the classifer analysis if necessary.
        print("Random Forest Classifier Report:")
        print(classification_report(y_test, y_predictions))
    return accuracy, rf_precision, rf_recall, rf_f1


def create_train_evaluate_RF(x, y) :
    random_forest = RandomForestClassifier(x, y)
    random_forest.fit(X_train, y_train)
    predictions = random_forest.predict(X_test)
    return evaluate_classifer(
        y_test, predictions, False)

# Exhausive tuning for different hyper parameters for Random Forest.
def tune_hyperparameters_for_RF(
        num_trees, max_depth, X_train, X_test, y_train, y_test) :
    best_num_tree = num_trees
    best_max_depth = max_depth
    numbers_of_tree = np.arange(best_num_tree, 1001, 2)
    max_depths = np.arange(best_max_depth, 20, 1)
    x, y = np.meshgrid(numbers_of_tree, max_depths)
    ufunc_hyper_tuning = np.frompyfunc(create_train_evaluate_RF, 2, 4)
    accuracy_list, rf_precision_list, rf_recall_list, rf_f1_list = ufunc_hyper_tuning(x, y)
    plot_3d_surface(x, y, accuracy_list)
    return best_num_tree, best_max_depth

if __name__=="__main__":
    dataframe_loaded = load_data_from_files()
    print(dataframe_loaded)
    dataframe_loaded = data_preprocess(dataframe_loaded)
    dataframe_scaled = feature_scaling(dataframe_loaded)
    dataframe_meaned = mean_filter(dataframe_scaled)
    features_list = feature_extraction(dataframe_meaned)
    
    # Convert the list to data frame to save in a csv file.
    feature_df = pd.DataFrame(features_list)
    feature_df.to_csv('feature_df.csv', sep=',', index=False, encoding='utf-8')
    
    # Split X from y to do data spliting.
    X = feature_df.iloc[:, :-1]
    y = feature_df.iloc[:, -1]
    
    # Data spliting
    X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert data to tensor type.
    X_train = torch.tensor(X_train_data.to_numpy(), dtype=torch.float32)
    X_test = torch.tensor(X_test_data.to_numpy(), dtype=torch.float32)
    y_train = torch.tensor(y_train_data.to_numpy(), dtype=torch.long)
    y_test = torch.tensor(y_test_data.to_numpy(), dtype=torch.long)
    
    # Create and train a Random Forest.
    # Make an initiall guess for number of trees and maximum depth.
    num_trees = 5
    max_depth = 5
    best_num_tree, best_max_depth = tune_hyperparameters_for_RF(
        num_trees, max_depth, X_train, X_test, y_train, y_test)
    
    print("Create RandomForestClassifer===========")
    random_forest = RandomForestClassifier(best_num_tree, best_max_depth)
    print("Fit RandomForestClassifer===========")
    random_forest.fit(X_train, y_train)
    print("RandomForestClassifer predicts()===========")
    rf_predictions = random_forest.predict(X_test)
    evaluate_classifer(y_test, rf_predictions, True)


#-----------------------------------------
# 定义交叉验证策略
#skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 对于随机森林和SVM，使用交叉验证计算准确度
#rf_cv_accuracy = cross_val_score(rf_classifier, encoded_features, y, cv=skf, scoring='accuracy')
#svm_cv_accuracy = cross_val_score(svm_classifier, encoded_features, y, cv=skf, scoring='accuracy')

#print(f"Random Forest CV Accuracy: {rf_cv_accuracy.mean()} (+/- {rf_cv_accuracy.std() * 2})")
#print(f"SVM CV Accuracy: {svm_cv_accuracy.mean()} (+/- {svm_cv_accuracy.std() * 2})")

# 定义一个简单的线性分类器模型
class LinearSVM(nn.Module):
    def __init__(self):
        super(LinearSVM, self).__init__()
        # 这里的10是输入特征的数量，3是类别的数量
        self.linear = nn.Linear(36, 3)

    def forward(self, x):
        return self.linear(x)


# 确定多分类Hinge损失
class MultiClassHingeLoss(nn.Module):
    def __init__(self):
        super(MultiClassHingeLoss, self).__init__()

    def forward(self, output, target):
        """
        output (batch_size, n_classes): 模型输出
        target (batch_size): 实际类别的索引
        """
        # 确定每个样本的正确分类的得分
        correct_class_scores = output[torch.arange(0, output.size(0)).long(), target].view(-1, 1)

        # 比较正确分类得分与其他分类得分的差距，并计算损失
        margin = 1.0
        loss = output - correct_class_scores + margin
        # 确保正确的分类不参与损失计算
        loss[torch.arange(0, output.size(0)).long(), target] = 0
        # 只取正的部分，相当于max(0, .)
        loss = torch.sum(torch.clamp(loss, min=0))
        return loss


def SVM_create() :
    # 初始化模型、损失函数和优化器
    model = LinearSVM()
    criterion = MultiClassHingeLoss()
    optimizer = SGD(model.parameters(), lr=0.01)
    
    
    
    # 训练模型
    for epoch in range(20):  # 训练20个epoch
        optimizer.zero_grad()  # 清除之前的梯度
        output = model(X_train)  # 前向传播
        loss = criterion(output, y_train)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
    
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
    
    # 模型评估
    with torch.no_grad():
        output = model(X_test)
        _, predicted = torch.max(output, 1)
        correct = (predicted == y_test).sum().item()
        print(f'Accuracy: {correct / len(y) * 100}%')