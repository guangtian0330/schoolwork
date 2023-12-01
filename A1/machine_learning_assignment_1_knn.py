import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import threading
import time


filename = ('D:\\lisis\Documents\\bank+marketing\\bank\\bank-full.csv')
Read = lambda filaname : pd.read_csv(filename, sep=";")

def data_preprocessing(df):
  X = df.drop('y', axis=1)
  y = df['y'].map({'yes': 1, 'no': 0})
  X['default'] = df['default'].map({'yes': 1, 'no': 0})
  X['housing'] = df['housing'].map({'yes': 1, 'no': 0})
  X['loan'] = df['loan'].map({'yes': 1, 'no': 0})
  categorical_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
  ct = ColumnTransformer(transformers=[('cat', OneHotEncoder(), categorical_cols)], remainder='passthrough')
  X_transformed = ct.fit_transform(X)
  return X_transformed, y


df = Read(filename)
X_data,y = data_preprocessing(df)
X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.2, random_state=42)




def run_knn(k, X_train, formula = 'Euclidean'):
  knn = KNN_classifier_with_distance_formula(X_train, k, formula)
  knn.fit(X_train, y_train)
  y_pred = knn.predict(X_test)
  return y_pred, knn

def KNN_classifier_with_distance_formula(X_train, k=3, distance_formula = 'Euclidean'):
    if distance_formula == 'Euclidean':
        return KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)
    elif distance_formula == 'Manhattan':
        return KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=1)
    elif distance_formula == 'Chebyshev':
        return KNeighborsClassifier(n_neighbors=k, metric='chebyshev')
    elif distance_formula == 'Mahalanobis':
        V = np.linalg.inv(np.cov(X_train, rowvar=False))
        return KNeighborsClassifier(n_neighbors=k, metric='mahalanobis', metric_params={'V': V})

def evaluation_model(y_pred):
  accuracy = accuracy_score(y_test, y_pred)
  print(f"Accuracy: {accuracy:.4f}")
  return accuracy

def plot_confusion_matrix(y_pred):
  cm = confusion_matrix(y_test, y_pred)
  plt.figure(figsize=(8, 6))
  sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.title('Confusion Matrix')
  plt.show()

# current_y_pred = run_knn(67)
# evaluation_model(current_y_pred)
# plot_confusion_matrix(current_y_pred)
#get_LogisticRegression()


'''
这里有个问题，在run_knn方法内部执行逻辑回归，和在方法外部执行逻辑回归，得到的Accuracy不一致。所以，需要研究如何给逻辑回归的方法传参。
-- Please Guangtian research it. <TODO>
'''


def run_knn_and_evaluate(formula):
    auc_values = []
    accuracy_values = []
    cross_val_scores = []

    # 取1-50的奇数
    ks = list(range(1, 100, 2))
    for k in ks:
        y_pred, knn = run_knn(k, X_train, formula)
        # 评估模型
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_values.append(accuracy)

        # 预测概率
        y_pred_prob = knn.predict_proba(X_test)[:, 1]
        # 计算ROC曲线和AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        auc = roc_auc_score(y_test, y_pred_prob)
        auc_values.append(auc)
        print("k: " + str(k) +"    "+ f"Accuracy: {accuracy:.4f}"+"    "+ f"AUC: {auc:.2f}")
        #交叉验证，搜集数据
        scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
        cross_val_scores.append(scores.mean())

          # 交叉验证，找到最佳k值
        best_k = ks[cross_val_scores.index(max(cross_val_scores))]
        print(f"The best value for k is: {best_k} in {formula}")


    plt.plot(ks, accuracy_values, color = "blue", marker='o', linestyle='-', label = "Accuracy Value")
    plt.title('Accuracy Rate vs K Value')
    plt.xlabel('K')
    plt.ylabel('Accuracy Rate')
    plt.show()

    plt.plot(ks, auc_values, color = "blue", marker='o', linestyle='-', label = "Accuracy Value")
    plt.title('AUC Rate vs K Value')
    plt.xlabel('K')
    plt.ylabel('AUC Rate')
    plt.show()

    plt.plot(ks, accuracy_values, color = "blue", marker='o', linestyle='-', label = "Accuracy Value")
    plt.plot(ks, auc_values, color = "red", marker='s', linestyle='--', label = "AUC_Value")
    plt.title('Rate vs K Value')
    plt.legend()
    plt.xlabel('K')
    plt.ylabel('Rate')
    plt.show()


distance_formula = ['Euclidean', 'Manhattan', 'Chebyshev']
thread_array = []
for index, formula in enumerate(distance_formula, start=1):
    thread = threading.Thread(target=run_knn_and_evaluate, args=(formula,))
    thread_array.append(thread)
    # 启动线程
    thread.start()

for e in thread_array:
    e.join()






# def get_AUC():
#   y_pred_prob = knn.predict_proba(X_test)[:, 1]
#   fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
#   auc = roc_auc_score(y_test, y_pred_prob)
#   print(f"AUC: {auc:.2f}")
#   return auc
#
# def plot_ROC_curve(auc):
#   plt.figure(figsize=(8, 6))
#   plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
#   plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#   plt.xlabel('False Positive Rate')
#   plt.ylabel('True Positive Rate')
#   plt.title('Receiver Operating Characteristic (ROC) Curve')
#   plt.legend(loc='lower right')
#   plt.show()

# auc_value = get_AUC()
# plot_ROC_curve(auc_value)


# def get_LogisticRegression(threshold = 0.5):
#   logreg = LogisticRegression(max_iter=5000)
#   logreg.fit(X_train, y_train)
#   probabilities = logreg.predict_proba(X_test)[:,1]
#   print("current threshold: " + str(threshold))
#   predictions = [1 if prob >= threshold else 0 for prob in probabilities]
#   print(confusion_matrix(y_test, predictions))
#   print(classification_report(y_test, predictions))
#   print(f"Logistic Regression Test Accuracy: {accuracy_score(y_test, y_pred)}")
#
#
# get_LogisticRegression()
# print("===================================================================")
# get_LogisticRegression(0.3)

