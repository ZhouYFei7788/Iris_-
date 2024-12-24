from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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
def to_tensor(data, targe):
    return torch.tensor(data, dtype=torch.float32), torch.tensor(targe, dtype=torch.long)

X_train_tensor, y_train_tensor = to_tensor(X_train, y_train)
X_test_tensor, y_test_tensor = to_tensor(X_test, y_test)

# 1. KNN算法
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("KNN准确率:", accuracy_score(y_test, y_pred_knn))

# 2. 逻辑回应算法
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
print("逻辑回应准确率:", accuracy_score(y_test, y_pred_log_reg))

# 3. 支持向量机 (SVM)
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("SVM准确率:", accuracy_score(y_test, y_pred_svm))

# 4. k-means聚类算法
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_train)
# 将k-means聚类结果映射到真实标签
kmeans_labels = np.zeros_like(kmeans.labels_)
for i in range(3):
    mask = (kmeans.labels_ == i)
    kmeans_labels[mask] = np.bincount(y_train[mask]).argmax()
y_pred_kmeans = kmeans.predict(X_test)
y_pred_kmeans = np.array([kmeans_labels[label] for label in y_pred_kmeans])
print("k-means准确率:", accuracy_score(y_test, y_pred_kmeans))

# 5. 朴素贝叶斯
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
print("朴素贝叶斯准确率:", accuracy_score(y_test, y_pred_nb))

# 6. 决策树
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("决策树准确率:", accuracy_score(y_test, y_pred_dt))

# 7. 随机森林
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("随机森林准确率:", accuracy_score(y_test, y_pred_rf))

# 8. AdaBoost
ada = AdaBoostClassifier(random_state=42)
ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)
print("AdaBoost准确率:", accuracy_score(y_test, y_pred_ada))

# 9. 梯度推动
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
print("梯度推动准确率:", accuracy_score(y_test, y_pred_gb))

# 10. 使用PyTorch实现神经网络
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return self.softmax(x)

# 定义模型、损失函数和优化器
input_size = X_train.shape[1]
hidden_size = 10
output_size = 3
model = NeuralNetwork(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
epochs = 50
for epoch in range(epochs):
    # 前向传播
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# 测试模型
with torch.no_grad():
    y_pred_nn = model(X_test_tensor)
    y_pred_nn = torch.argmax(y_pred_nn, axis=1)
    print("神经网络准确率 (PyTorch):", accuracy_score(y_test_tensor, y_pred_nn))
