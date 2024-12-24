from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 加载鹦尾花数据集
iris = load_iris()
X, y = iris.data, iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 标准化数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 转换为PyTorch的Tensor
def to_tensor(data, target):
    return torch.tensor(data, dtype=torch.float32), torch.tensor(target, dtype=torch.long)


X_train_tensor, y_train_tensor = to_tensor(X_train, y_train)
X_test_tensor, y_test_tensor = to_tensor(X_test, y_test)


# 计算并打印各模型的准确率、AUC、Recall和F1值
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    try:
        auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
    except AttributeError:
        auc = 'AUC无法估算'

    return accuracy, auc, recall, f1


# 1. KNN算法
knn = KNeighborsClassifier(n_neighbors=3)
accuracy_knn, auc_knn, recall_knn, f1_knn = evaluate_model(knn, X_train, y_train, X_test, y_test)
print(f"KNN准确率: {accuracy_knn:.4f}, AUC: {auc_knn}, Recall: {recall_knn:.4f}, F1: {f1_knn:.4f}")

# 2. 逻辑回归
log_reg = LogisticRegression()
accuracy_log_reg, auc_log_reg, recall_log_reg, f1_log_reg = evaluate_model(log_reg, X_train, y_train, X_test, y_test)
print(f"逻辑回归准确率: {accuracy_log_reg:.4f}, AUC: {auc_log_reg}, Recall: {recall_log_reg:.4f}, F1: {f1_log_reg:.4f}")

# 3. 支持向量机 (SVM)
svm = SVC(kernel='linear', probability=True)
accuracy_svm, auc_svm, recall_svm, f1_svm = evaluate_model(svm, X_train, y_train, X_test, y_test)
print(f"SVM准确率: {accuracy_svm:.4f}, AUC: {auc_svm}, Recall: {recall_svm:.4f}, F1: {f1_svm:.4f}")

# 4. k-means聚类算法
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_train)
kmeans_labels = np.zeros_like(kmeans.labels_)
for i in range(3):
    mask = (kmeans.labels_ == i)
    kmeans_labels[mask] = np.bincount(y_train[mask]).argmax()
y_pred_kmeans = kmeans.predict(X_test)
y_pred_kmeans = np.array([kmeans_labels[label] for label in y_pred_kmeans])
accuracy_kmeans = accuracy_score(y_test, y_pred_kmeans)
recall_kmeans = recall_score(y_test, y_pred_kmeans, average='macro')
f1_kmeans = f1_score(y_test, y_pred_kmeans, average='macro')
print(f"k-means准确率: {accuracy_kmeans:.4f}, Recall: {recall_kmeans:.4f}, F1: {f1_kmeans:.4f}")

# 5. 朴素贝叶斯
nb = GaussianNB()
accuracy_nb, auc_nb, recall_nb, f1_nb = evaluate_model(nb, X_train, y_train, X_test, y_test)
print(f"朴素贝叶斯准确率: {accuracy_nb:.4f}, AUC: {auc_nb}, Recall: {recall_nb:.4f}, F1: {f1_nb:.4f}")

# 6. 决策树
dt = DecisionTreeClassifier(random_state=42)
accuracy_dt, auc_dt, recall_dt, f1_dt = evaluate_model(dt, X_train, y_train, X_test, y_test)
print(f"决策树准确率: {accuracy_dt:.4f}, AUC: {auc_dt}, Recall: {recall_dt:.4f}, F1: {f1_dt:.4f}")

# 7. 随机森林
rf = RandomForestClassifier(random_state=42)
accuracy_rf, auc_rf, recall_rf, f1_rf = evaluate_model(rf, X_train, y_train, X_test, y_test)
print(f"随机森林准确率: {accuracy_rf:.4f}, AUC: {auc_rf}, Recall: {recall_rf:.4f}, F1: {f1_rf:.4f}")

# 8. AdaBoost
ada = AdaBoostClassifier(random_state=42)
accuracy_ada, auc_ada, recall_ada, f1_ada = evaluate_model(ada, X_train, y_train, X_test, y_test)
print(f"AdaBoost准确率: {accuracy_ada:.4f}, AUC: {auc_ada}, Recall: {recall_ada:.4f}, F1: {f1_ada:.4f}")

# 9. 梯度推动
gb = GradientBoostingClassifier(random_state=42)
accuracy_gb, auc_gb, recall_gb, f1_gb = evaluate_model(gb, X_train, y_train, X_test, y_test)
print(f"梯度推动准确率: {accuracy_gb:.4f}, AUC: {auc_gb}, Recall: {recall_gb:.4f}, F1: {f1_gb:.4f}")
# # 10. 全连接神经网络
# class NeuralNetwork(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(NeuralNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, output_size)
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
#         return self.softmax(x)
#
# # 定义模型、损失函数和优化器
# input_size = X_train.shape[1]
# hidden_size = 10
# output_size = 3
# model = NeuralNetwork(input_size, hidden_size, output_size)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
#
# # 训练模型
# epochs = 50
# for epoch in range(epochs):
#     # 前向传播
#     outputs = model(X_train_tensor)
#     loss = criterion(outputs, y_train_tensor)
#
#     # 反向传播和优化
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     if (epoch + 1) % 10 == 0:
#         print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
#
# # 测试模型
# with torch.no_grad():
#     y_pred_nn = model(X_test_tensor)
#     y_pred_nn = torch.argmax(y_pred_nn, axis=1)
#     print("神经网络准确率 (PyTorch):", accuracy_score(y_test_tensor, y_pred_nn))
