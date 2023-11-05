from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 加载数据集-------------------------------------
jumping_jack_accelerometer = pd.read_csv('./Jumping_Jack_x10/Accelerometer.csv', sep=',')
jumping_jack_orientation = pd.read_csv('./Jumping_Jack_x10/Orientation.csv', sep=',')
jumping_jack_total_orientation = pd.read_csv('./Jumping_Jack_x10/TotalAcceleration.csv', sep=',')

lunges_accelerometer = pd.read_csv('./Lunges_x10/Accelerometer.csv', sep=',')
lunges_orientation = pd.read_csv('./Lunges_x10/Orientation.csv', sep=',')
lunges_total_orientation = pd.read_csv('./Lunges_x10/TotalAcceleration.csv', sep=',')

squat_accelerometer = pd.read_csv('./Squat_x10/Accelerometer.csv', sep=',')
squat_orientation = pd.read_csv('./Squat_x10/Orientation.csv', sep=',')
squat_total_orientation = pd.read_csv('./Squat_x10/TotalAcceleration.csv', sep=',')
# 加载数据集-------------------------------------


# 设置窗口大小
window_size = 100
# 提取特征的函数
def extract_features(window):
    features = {
        'mean_x': window['x'].mean(),
        'std_x': window['x'].std(),
        'max_x': window['x'].max(),
        'min_x': window['x'].min(),
        'mean_y': window['y'].mean(),
        'std_y': window['y'].std(),
        'max_y': window['y'].max(),
        'min_y': window['y'].min(),
        'mean_z': window['z'].mean(),
        'std_z': window['z'].std(),
        'max_z': window['z'].max(),
        'min_z': window['z'].min(),
    }
    return features

# 划分窗口并提取特征
feature_list = []
for i in range(0, len(jumping_jack_accelerometer) - window_size + 1):
    window = jumping_jack_accelerometer[i:i+window_size]
    feature_list.append(extract_features(window))

# 转换为DataFrame
feature_data = pd.DataFrame(feature_list)

jumping_jack_feature_vectors = [extract_features(jumping_jack_accelerometer[i:i + window_size]) for i in range(0, len(jumping_jack_accelerometer), window_size)]
lunges_feature_vectors = [extract_features(lunges_accelerometer[i:i + window_size]) for i in range(0, len(lunges_accelerometer), window_size)]
squat_vectors = [extract_features(squat_accelerometer[i:i + window_size]) for i in range(0, len(squat_accelerometer), window_size)]


# 创建标签
labels_action1 = [0 for _ in range(len(jumping_jack_feature_vectors))]  # 假设第一组动作的标签为0
labels_action2 = [1 for _ in range(len(lunges_feature_vectors))]  # 第二组动作的标签为1
labels_action3 = [2 for _ in range(len(squat_vectors))]  # 第三组动作的标签为2
X_train, X_test = train_test_split(feature_data, test_size=0.3, random_state=42)


# 合并特征向量和标签
features = jumping_jack_feature_vectors + lunges_feature_vectors + squat_vectors
labels = labels_action1 + labels_action2 + labels_action3

# 转换为适合机器学习的格式
features = pd.DataFrame(features)
labels = pd.Series(labels)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)


# 创建SVM分类器实例
svm_classifier = SVC(kernel='linear')  # 你可以选择不同的核函数

# 训练模型
svm_classifier.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = svm_classifier.predict(X_test)
# Generate a detailed classification report showing performance metrics for the SVM.
svm_classification_report = classification_report(y_test, y_pred)

# 评估模型
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("SVM Classification Report:\n", svm_classification_report)



# Random Forest Classifier: An ensemble of decision trees.
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # Initialize the Random Forest with 100 trees and a fixed random state for reproducibility.
rf_model.fit(X_train, y_train)  # Train the Random Forest model on the training data.
rf_predictions = rf_model.predict(X_test)  # Use the trained model to predict the labels of the test data.

# Calculate the accuracy of the Random Forest model's predictions.
rf_accuracy = accuracy_score(y_test, rf_predictions)

# Generate a detailed classification report showing performance metrics for the Random Forest.
rf_classification_report = classification_report(y_test, rf_predictions)

# Display the Random Forest's accuracy and classification report.
print("Random Forest Accuracy:", rf_accuracy)
print("Random Forest Classification Report:\n", rf_classification_report)