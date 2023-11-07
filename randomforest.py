import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载示例数据集
data = load_iris()
X = torch.tensor(data.data, dtype=torch.float32)
y = torch.tensor(data.target, dtype=torch.long)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# 定义一个简单的决策树模型
class DecisionTree(nn.Module):
    def __init__(self, max_depth):
        super(DecisionTree, self).__init__()
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y, depth=0):
        if depth >= self.max_depth:
            return {"class": torch.bincount(y).argmax()}
        
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
                gini = (len(left) / num_samples) * self.gini_impurity(left) + (len(right) / num_samples) * self.gini_impurity(right)
                if gini < best_gini:
                    best_gini = gini
                    best_split = (feature_index, value)
                    left_X, right_X, left_y, right_y = X[split_mask], X[~split_mask], y[split_mask], y[~split_mask]

        if best_gini == 1.0:
            return {"class": torch.bincount(y).argmax()}

        self.tree = {
            "feature_index": best_split[0],
            "split_value": best_split[1],
            "left": self.fit(left_X, left_y, depth + 1),
            "right": self.fit(right_X, right_y, depth + 1)
        }
        return self.tree

    def gini_impurity(self, y):
        if len(y) == 0:
            return 0
        p = torch.bincount(y) / len(y)
        return 1 - torch.sum(p ** 2)

    def predict(self, X):
        if self.tree is None:
            raise Exception("The tree is not trained yet.")
        return torch.tensor([self._predict(x, self.tree) for x in X])

    def _predict(self, x, node):
        if "split_value" not in node:
            return node["class"]  # 叶子节点返回类别标签或回归值
        if x[node["feature_index"]] <= node["split_value"]:
            return self._predict(x, node["left"])
        else:
            return self._predict(x, node["right"])

# 创建并训练随机森林（多个决策树）
class RandomForestClassifier:
    def __init__(self, num_trees, max_depth):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.trees = [DecisionTree(max_depth) for _ in range(num_trees)]
    
    def fit(self, X, y):
        for tree in self.trees:
            tree.fit(X, y)
    
    def predict(self, X):
        predictions = [tree.predict(X) for tree in self.trees]
        return torch.stack(predictions, dim=0).mode(0).values

# 创建并训练随机森林分类器
num_trees = 8
max_depth = 5
random_forest = RandomForestClassifier(num_trees, max_depth)
random_forest.fit(X_train, y_train)

# 预测
y_pred = random_forest.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)