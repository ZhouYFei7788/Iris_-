import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)


# 3. 定义全连接神经网络模型
class IrisFCNN(nn.Module):
    def __init__(self):
        super(IrisFCNN, self).__init__()
        self.fc1 = nn.Linear(4, 16)  # 输入层（4个特征），隐藏层（16个神经元）
        self.fc2 = nn.Linear(16, 8)  # 隐藏层（16个神经元），输出层（8个神经元）
        self.fc3 = nn.Linear(8, 3)  # 输出层（3类）
        self.relu = nn.ReLU()  # 激活函数

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 4. 初始化模型、损失函数和优化器
model = IrisFCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 5. 训练模型
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # 前向传播
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # 反向传播和优化
    loss.backward()
    optimizer.step()

    # 每10个epoch打印一次损失
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# 6. 测试模型
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    acc = accuracy_score(y_test, predicted)

print(f"Test Accuracy: {acc * 100:.2f}%")

# # 7.保存模型
# torch.save(model.state_dict(), "model.pth")