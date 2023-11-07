import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from scipy.stats import skew, kurtosis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report


# 数据加载  ./Jumping_Jack_x10/Accelerometer
motions = ['./Jumping_Jack_x10', './Lunges_x10', './Squat_x10']
sensors = ['/Accelerometer', '/Orientation', '/TotalAcceleration']
# sensors = ['/TotalAcceleration']
dataframes = {}

for motion in motions:
    for sensor in sensors:
        filename = f"{motion}{sensor}.csv"
        df_key = f"{motion}{sensor}"
        dataframes[df_key] = pd.read_csv(filename)

print("====================数据加载 Done")


# 数据清洗
for df_key, df in dataframes.items():
    # 计算原始 NaN 值的数量
    original_nan_count = df.isna().sum().sum()

    # 去重复
    df.drop_duplicates(inplace=True)
    # 替换无穷大值为NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # 前向填充处理一部分 NaN 值
    df.ffill(inplace=True)

    # 计算前向填充后 NaN 值的数量
    nan_count_after_ffill = df.isna().sum().sum()

    # 创建一个 imputer 对象，用列的均值填充剩余的 NaN 值
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    # 应用 imputer 到每个 dataframe
    df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # 计算填充均值后 NaN 值的数量
    nan_count_after_imputing = df_filled.isna().sum().sum()

    # 打印信息
    print(f"{df_key}:")
    print(f"Original NaN count: {original_nan_count}")
    print(f"NaN count after forward fill: {nan_count_after_ffill}")
    print(f"NaN filled with forward fill: {original_nan_count - nan_count_after_ffill}")
    print(f"NaN count after mean imputation: {nan_count_after_imputing}")
    print(f"NaN filled with mean imputation: {nan_count_after_ffill - nan_count_after_imputing}")

    # 更新字典中的DataFrame
    dataframes[df_key] = df_filled


print("====================数据清洗 Done")

# 处添加噪声 好奇怪 貌似没有起作用 还是过拟合
noise_level = 0.01  # 噪声水平可以根据需要调整
for df_key, df in dataframes.items():
    for col in df.columns:
        if col not in ['motion', 'sensor']:  # 仅对数值特征添加噪声
            df[col] = df[col] + np.random.normal(0, noise_level, size=df[col].shape)


'''
去除 'time' 和 'seconds_elapsed' 这两个特征通常是基于以下几个考虑：

不携带预测性信息：如果时间戳本身与目标变量（在您的案例中是运动类型）无关，则它们不会为模型提供任何有价值的信息。比如，如果运动类型不随时间变化而变化，那么时间特征可能对模型预测不会有帮助。

数据泄露：如果时间相关的特征与数据收集的方式有关，而不是与预测目标本身有关，那么它们可能会导致数据泄露，使模型过度拟合训练集中的“时间”模式，而这些模式在未来的数据中可能并不适用。

模型复杂性：保留不相关或不重要的特征会增加模型的复杂性，可能导致模型训练时间更长，而且如果特征过多还可能导致维度灾难。

时间序列问题：如果数据是时间序列数据，'time' 和 'seconds_elapsed' 可能需要以不同的方式处理，如通过时间序列分析而不是标准的机器学习模型。对于这种数据，通常会使用特殊的时间序列模型或将时间信息转换为其他更有用的特征。

在一些情况下，时间特征可以经过适当的工程转换为模型可用的形式。例如，如果您知道某些运动类型在一天中的特定时间更有可能发生，您可以将时间戳转换为一天中的小时来捕获这种周期性。或者，如果您有理由相信运动类型会随着时间推移而发生变化（例如，在长时间的数据收集期间），那么时间也可能成为一个重要特征。
'''
# 均值滤波
for df_key, df in dataframes.items():
    for col in ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'roll', 'pitch', 'yaw']:
        if col in df.columns:
            df[col] = df[col].rolling(window=5, min_periods=1).mean()

print("====================均值滤波 Done")

# 滑动窗口划分和特征提取
window_size = 20  # 定义窗口大小
step_size = 10    # 定义步长
features_list = []

for motion in motions:
    for sensor in sensors:
        df_key = f"{motion}{sensor}"
        df = dataframes[df_key]

        # 根据传感器类型选择列
        if sensor == '/Orientation':
            # columns = ['qz', 'qy', 'qx', 'qw', 'roll', 'pitch', 'yaw']
            '''
            方向的 qz, qy, qx, qw（四元数），以及 roll, pitch, yaw：
            四元数和欧拉角（roll, pitch, yaw）都提供了关于设备在空间中方向的信息。
            通常不需要这两种类型的方向数据，因为它们表示相同的信息。
            四元数不会受到万向锁的影响，因此更稳定，但欧拉角更直观和易于理解。
            '''
            columns = ['roll', 'pitch', 'yaw']
        else:  # 'Accelerometer' or 'TotalAcceleration'
            columns = ['x', 'y', 'z']

        for start in range(0, df.shape[0] - window_size + 1, step_size):
            end = start + window_size
            window_df = df.iloc[start:end]
            feature = {'motion': motion, 'sensor': sensor}

            # 计算每个选定列的统计特征
            for col in columns:
                feature[f'mean_{col}'] = window_df[col].mean()
                feature[f'std_{col}'] = window_df[col].std()
                feature[f'skew_{col}'] = skew(window_df[col], nan_policy='omit')
                feature[f'kurt_{col}'] = kurtosis(window_df[col], nan_policy='omit')
                feature[f'max_{col}'] = window_df[col].max()
                feature[f'min_{col}'] = window_df[col].min()

            # 将特征添加到列表中
            features_list.append(feature)

print("====================滑动窗口划分和特征提取 Done")

# 将特征列表转换为DataFrame
feature_df = pd.DataFrame(features_list)

feature_df.to_csv('feature_df.csv', sep='\t', index=False, encoding='utf-8')

# 这会将DataFrame df保存为一个以制表符分隔的文本文件，不包含索引。

print("====================将特征列表转换为DataFrame Done")
'''
关于列名不同，导致了特征值出现空值的问题。P哥给的建议如下： 

数据对齐：您应该确保在合并不同数据源的特征前，所有特征都能对齐。例如，如果某个特征仅在某些传感器中存在，您可以在其他传感器的特征集中添加该特征并填充默认值（如0或NaN）。这将保持特征集的大小一致，同时保留了哪些传感器提供了哪些数据的信息。

自定义填充策略：针对每种传感器数据特征的缺失值，您可以定义一个合理的填充策略。例如，对于Orientation传感器数据中不存在的x, y, z列，您可以考虑填充为0，因为这可能表示该方向上的加速度或旋转角度为零。但请注意，这种策略应根据实际物理含义谨慎选择。

特征工程：您可以创建更复杂的特征，比如基于现有数据计算出新的统计量或者结合不同传感器的数据计算出交叉特征。例如，您可以使用Orientation和Accelerometer数据计算出某种动态行为模式的特征。

多模型方法：构建多个模型，每个模型专注于不同传感器的数据。最后，您可以通过集成学习方法（如投票、堆叠或融合）来结合这些模型的预测，从而进行最终的运动类型识别。

特征选择：如果某些特征在合并后通常是缺失的，可能表明这些特征对于某些运动类型的识别并不重要。您可以考虑仅使用在所有传感器数据中一致出现的特征。

异常值处理：在处理特征之前，先进行异常值检测，确保所有的特征都在合理的范围内。如果检测到异常值，可以适当处理（例如，通过替换为中位数或均值）。

模型选择：有些机器学习模型可以处理缺失值，例如XGBoost或LightGBM。您可以选择这样的模型来避免预处理阶段的复杂填充策略。

数据增强：生成额外的数据来填充缺失值，这可以通过模拟或其他数据生成技术来完成，但需要确保生成的数据在统计上与真实数据相似。
'''

# 移除非数值列
feature_numeric = feature_df.drop(['motion', 'sensor'], axis=1)

# 空值填充：使用SimpleImputer填充均值
# imputer = SimpleImputer(strategy='mean')

# 空值填充：使用SimpleImputer填充0
imputer = SimpleImputer(strategy='constant', fill_value=0)

feature_numeric_imputed = imputer.fit_transform(feature_numeric)

# 特征标准化：对数值特征进行标准化
scaler = StandardScaler()
feature_numeric_scaled = scaler.fit_transform(feature_numeric_imputed)

# 对类别特征进行处理（目标分类）
# 提取类别特征
feature_categorical = feature_df[['motion']]

# 使用OneHotEncoder进行类别特征的独热编码
encoder = OneHotEncoder(sparse_output=False)
feature_categorical_encoded = encoder.fit_transform(feature_categorical)

# 最后，合并数值特征和编码后的类别特征
encoded_features = np.concatenate([feature_numeric_scaled, feature_categorical_encoded], axis=1)

print("现在数据长这样，好丑：")
print(encoded_features[:3])


# print(encoded_features.to_csv('encoded_features.csv', sep='\t', index=False, encoding='utf-8'))sparse

print("====================十八般武艺的特征处理 Done")


y = feature_df['motion'].values  # 提取目标变量

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(encoded_features, y, test_size=0.2, random_state=42)
print("切割后的数据：X_train")
print(X_train[:3])

'''

接下来的代码讲道理应该使用pytorch
然而我困了
瞎写着玩
过拟合了
再见

'''
# 训练随机森林模型

# 调整随机森林模型的参数 解决过拟合问题
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    max_depth=None,  # 或者设置为一个合理的值，以限制决策树的深度
    random_state=42
)
rf_classifier.fit(X_train, y_train)


# 训练SVM模型
svm_classifier = SVC(
    C=0.5,  # 减少C的值增加正则化的强度
    kernel='rbf',  # RBF核通常是一个好的选择，但也可以尝试线性或多项式核
    random_state=42
)
svm_classifier.fit(X_train, y_train)

# 预测测试集
rf_predictions = rf_classifier.predict(X_test)
svm_predictions = svm_classifier.predict(X_test)

# 评估模型性能
print("Random Forest Classifier:")
print("Accuracy:", accuracy_score(y_test, rf_predictions))

print("\nSVM Classifier:")
print("Accuracy:", accuracy_score(y_test, svm_predictions))


# 定义交叉验证策略
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 对于随机森林和SVM，使用交叉验证计算准确度
rf_cv_accuracy = cross_val_score(rf_classifier, encoded_features, y, cv=skf, scoring='accuracy')
svm_cv_accuracy = cross_val_score(svm_classifier, encoded_features, y, cv=skf, scoring='accuracy')

print(f"Random Forest CV Accuracy: {rf_cv_accuracy.mean()} (+/- {rf_cv_accuracy.std() * 2})")
print(f"SVM CV Accuracy: {svm_cv_accuracy.mean()} (+/- {svm_cv_accuracy.std() * 2})")

# 混淆矩阵
rf_confusion_matrix = confusion_matrix(y_test, rf_predictions)
svm_confusion_matrix = confusion_matrix(y_test, svm_predictions)

# 精确度、召回率和F1分数
rf_precision = precision_score(y_test, rf_predictions, average='weighted')
rf_recall = recall_score(y_test, rf_predictions, average='weighted')
rf_f1 = f1_score(y_test, rf_predictions, average='weighted')

svm_precision = precision_score(y_test, svm_predictions, average='weighted')
svm_recall = recall_score(y_test, svm_predictions, average='weighted')
svm_f1 = f1_score(y_test, svm_predictions, average='weighted')

# 打印评估指标
print(f"Random Forest Confusion Matrix:\n{rf_confusion_matrix}")
print(f"Random Forest Precision: {rf_precision}")
print(f"Random Forest Recall: {rf_recall}")
print(f"Random Forest F1 Score: {rf_f1}")

print(f"SVM Confusion Matrix:\n{svm_confusion_matrix}")
print(f"SVM Precision: {svm_precision}")
print(f"SVM Recall: {svm_recall}")
print(f"SVM F1 Score: {svm_f1}")

# 分类报告
print("Random Forest Classifier Report:")
print(classification_report(y_test, rf_predictions))

print("\nSVM Classifier Report:")
print(classification_report(y_test, svm_predictions))
