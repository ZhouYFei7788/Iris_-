import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D 可视化所需
import plotly.express as px
import plotly.graph_objects as go

# 1. 加载数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# 2. 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 降维算法
# PCA降维
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# TSVD降维
tsvd = TruncatedSVD(n_components=3)
X_tsvd = tsvd.fit_transform(X_scaled)

# LDA降维
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

# t-SNE降维 (优化过的参数)
# 修改 t-SNE 参数
tsne = TSNE(n_components=3, random_state=42, perplexity=5, max_iter=250, n_jobs=-1)

X_tsne = tsne.fit_transform(X_scaled)

# 4. 二维和三维动态可视化
def plot_2d(X, y, method_name):
    plt.figure(figsize=(8, 6))
    for i, target_name in enumerate(target_names):
        plt.scatter(X[y == i, 0], X[y == i, 1], label=target_name)
    plt.title(f'2D Visualization ({method_name})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.show()

def plot_3d_matplotlib(X, y, method_name):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    for i, target_name in enumerate(target_names):
        ax.scatter(X[y == i, 0], X[y == i, 1], X[y == i, 2], label=target_name)
    ax.set_title(f'3D Visualization ({method_name})')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.legend()
    plt.show()

def plot_3d_plotly(X, y, method_name):
    fig = px.scatter_3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], color=y.astype(str),
                        labels={'color': 'Class'}, title=f'3D Visualization ({method_name})')
    fig.write_html("plotly_output.html")  # 将图保存为HTML文件
    fig.show()

# 二维可视化 (PCA, TSVD, LDA)
plot_2d(X_pca[:, :2], y, 'PCA')
plot_2d(X_tsvd[:, :2], y, 'TSVD')
plot_2d(X_lda, y, 'LDA')

# 三维可视化 (PCA, TSVD, t-SNE)
plot_3d_matplotlib(X_pca, y, 'PCA')  # 使用 Matplotlib 显示
plot_3d_matplotlib(X_tsvd, y, 'TSVD')  # 使用 Matplotlib 显示
plot_3d_matplotlib(X_tsne, y, 't-SNE')  # 使用 Matplotlib 显示

# 如果想使用 Plotly 可视化 (可选)
plot_3d_plotly(X_pca, y, 'PCA')
# plot_3d_plotly(X_tsvd, y, 'TSVD')
# plot_3d_plotly(X_tsne, y, 't-SNE')
