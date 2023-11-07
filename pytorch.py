# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 17:45:05 2023

@author: tian
"""
import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

param_grid = {
    # Normalizing parameter C
    'C': [0.1, 1, 10],
    # The kernel function for SVM.
    'kernel': ['linear', 'rbf', 'poly'],
    # The gamma value used in SVM.
    'gamma': [0.001, 0.01, 0.1]  # 不同的gamma值
}


class SVM(nn.Module):
    def __init__(self, input_dim, C, kernel, gamma):
        super(SVM, self).__init__()
        self.fc = nn.Linear(input_dim, 3)  # 3类输出
        self.C = C
        self.kernel = kernel
        self.gamma = gamma

    def forward(self, x):
        return self.fc(x)
    
# 迭代超参数组合进行训练和验证
best_accuracy = 0.0
best_params = None

def hinge_loss(scores, targets):
    loss = torch.mean(F.relu(1 - scores * targets))  # Hinge Loss
    return loss

# 准备训练数据和标签
X_train = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
y_train = torch.tensor([1, 1, -1, -1], dtype=torch.float32)

# 创建SVM模型
for C in param_grid['C']:
    for kernel in param_grid['kernel']:
        for gamma in param_grid['gamma']:
            model = SVM(input_dim=X_train.shape[1], C=C, kernel=kernel, gamma=gamma)
            optimizer = optim.SGD(model.parameters(), lr=0.01)

            for epoch in range(100):
                optimizer.zero_grad()
                outputs = model(X_train)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                model.eval()
                outputs_val = model(X_val)
                _, predicted = torch.max(outputs_val, 1)
                accuracy = accuracy_score(y_val, predicted)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {'C': C, 'kernel': kernel, 'gamma': gamma}

print("Best Hyperparameters:", best_params)
# 定义优化器
optimizer = torch.optim.SGD(svm_model.parameters(), lr=0.01)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    # 前向传播
    scores = svm_model(X_train)

    # 计算损失
    loss = hinge_loss(scores, y_train)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印损失
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

# 使用训练好的模型进行预测
with torch.no_grad():
    predicted_labels = torch.sign(svm_model(X_train))
    print("Predicted Labels:", predicted_labels)






# 创建一个简单的决策树模型
class DecisionTree(nn.Module):
    def __init__(self):
        super(DecisionTree, self).__init()
        self.tree = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.tree(x)

# 创建一个随机森林模型
class RandomForest(nn.Module):
    def __init__(self, num_trees):
        super(RandomForest, self).__init()
        self.num_trees = num_trees
        self.trees = nn.ModuleList([DecisionTree() for _ in range(num_trees)])

    def forward(self, x):
        outputs = [tree(x) for tree in self.trees]
        return torch.cat(outputs, dim=1)

# 准备数据
X_train = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
y_train = torch.tensor([1, 1, -1, -1], dtype=torch.float32)

# 创建RF模型
rf_model = RandomForest(num_trees=100)
optimizer = optim.SGD(rf_model.parameters(), lr=0.01)

# 训练RF模型
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    scores = rf_model(X_train)
    loss = torch.mean(torch.relu(1 - scores * y_train.view(-1, 1)))
    loss.backward()
    optimizer.step()

# 使用训练好的RF模型进行预测
with torch.no_grad():
    predicted_scores = rf_model(X_train)
    predicted_labels = torch.sign(predicted_scores)
    print("Predicted Labels:", predicted_labels)