import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import cx_Oracle
from yellowbrick.classifier import ClassPredictionError
#################################################此版弃用################################################################
#################################################此版弃用################################################################
#################################################此版弃用################################################################
#################################################此版弃用################################################################
#################################################此版弃用################################################################
#################################################此版弃用################################################################
#################################################此版弃用################################################################
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial']  # 如果没有'Microsoft YaHei'，可以使用系统其他字体

# 加载鸢尾花数据集
iris = load_iris()
X, y = iris.data, iris.target

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# 模型评估函数
def evaluate_model(model, X, y):
    # 使用交叉验证，计算模型的准确率
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')  # cv=5表示5折交叉验证
    return scores.mean(), scores.std()


# 连接数据库
def connect_to_db():
    dsn_tns = cx_Oracle.makedsn('localhost', '1521', service_name='XE')  # 修改为实际的主机和服务名
    connection = cx_Oracle.connect(user='C##YUANWEIHUA', password='root', dsn=dsn_tns)  # 修改为实际的用户名和密码
    return connection


# 创建数据库连接
conn = cx_Oracle.connect('C##YUANWEIHUA/root@XE')

# 创建游标
cursor = conn.cursor()


def insert_model_results(model_name, mean_score, std_score, epoch_loss=None, epoch=None):
    cursor = conn.cursor()

    # 获取最大记录ID并加1
    cursor.execute("SELECT MAX(记录ID) FROM 综合信息")
    result = cursor.fetchone()
    next_record_id = result[0] + 1 if result[0] is not None else 1

    # 使用双引号括起中文列名
    if epoch_loss is not None and epoch is not None:  # 如果是每轮的损失数据
        sql = """
            INSERT INTO 综合信息 (记录ID, 模型名称, 评估平均准确率, "评估准确率标准差", "训练轮数", "训练损失") 
            VALUES (:record_id, :model_name, :mean_score, :std_score, :epoch, :epoch_loss)
        """
        cursor.execute(sql, record_id=next_record_id, model_name=model_name, mean_score=mean_score, std_score=std_score,
                       epoch=epoch, epoch_loss=epoch_loss)
    else:
        sql = """
            INSERT INTO 综合信息 (记录ID, 模型名称, 评估平均准确率, "评估准确率标准差") 
            VALUES (:record_id, :model_name, :mean_score, :std_score)
        """
        cursor.execute(sql, record_id=next_record_id, model_name=model_name, mean_score=mean_score, std_score=std_score)

    conn.commit()


# 模拟模型结果
model_name = "KNN"
mean_score = 0.9533
std_score = 0.0340

# 训练并评估多个模型
models = [
    ("KNN", KNeighborsClassifier(n_neighbors=3)),
    ("逻辑回归", LogisticRegression(max_iter=200)),
    ("支持向量机", SVC(kernel='linear')),
    ("K-means聚类", KMeans(n_clusters=3, random_state=42)),
    ("朴素贝叶斯", GaussianNB()),
    ("决策树", DecisionTreeClassifier(random_state=42)),
    ("随机森林", RandomForestClassifier(random_state=42)),
    ("AdaBoost", AdaBoostClassifier(random_state=42)),
    ("梯度提升", GradientBoostingClassifier(random_state=42))
]

model_names = []
model_scores = []
model_stds = []

for name, model in models:
    print(f"\n{name}模型:")
    mean_score, std_score = evaluate_model(model, X, y)
    model_names.append(name)
    model_scores.append(mean_score)
    model_stds.append(std_score)
    print(f"交叉验证准确率: {mean_score:.4f} ± {std_score:.4f}")

    # 将模型评估结果插入数据库
    insert_model_results(name, mean_score, std_score)

# 可视化各模型的交叉验证准确率
plt.figure(figsize=(10, 6))
plt.barh(model_names, model_scores, xerr=model_stds, color='skyblue')
plt.title("各模型交叉验证准确率")
plt.xlabel("准确率")
plt.ylabel("模型")
plt.show()

# 可视化分类器的错误预测 (以KNN为例)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
visualizer = ClassPredictionError(knn)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()


# 使用PyTorch训练神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def cross_val_pytorch(model_class, X, y, cv=5, epochs=10):
    fold_size = len(X) // cv
    accuracies = []
    losses = []

    for i in range(cv):
        # 划分训练和验证集
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i != cv - 1 else len(X)

        X_train_fold = torch.cat([X[:val_start], X[val_end:]])
        y_train_fold = torch.cat([y[:val_start], y[val_end:]])
        X_val_fold = X[val_start:val_end]
        y_val_fold = y[val_start:val_end]

        # 模型训练
        model = model_class(X_train_fold.shape[1], 10, 3)  # 假设输入大小为特征数，输出为3
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        epoch_losses = []
        # 训练
        for epoch in range(epochs):
            outputs = model(X_train_fold)
            loss = criterion(outputs, y_train_fold)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

            # 每轮训练后，将损失写入数据库
            insert_model_results("神经网络", np.mean(accuracies), np.std(accuracies), epoch_loss=loss.item(),
                                 epoch=epoch + 1)

        # 记录每个fold的损失值
        losses.append(epoch_losses)

        # 验证
        with torch.no_grad():
            y_pred_val = model(X_val_fold)
            y_pred_val = torch.argmax(y_pred_val, dim=1)
            accuracy = accuracy_score(y_val_fold, y_pred_val)
            accuracies.append(accuracy)

    print(f"神经网络交叉验证准确率: {np.mean(accuracies) * 100:.2f}% ± {np.std(accuracies) * 100:.2f}%")

    # 可视化损失曲线
    avg_losses = np.mean(losses, axis=0)
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), avg_losses, label="训练损失")
    plt.title("神经网络训练过程的损失曲线")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


# 对神经网络使用交叉验证
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
cross_val_pytorch(NeuralNetwork, X_train_tensor, y_train_tensor)

# 关闭数据库连接
conn.close()
