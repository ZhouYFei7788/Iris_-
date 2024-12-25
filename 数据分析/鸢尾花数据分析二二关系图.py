import seaborn as sns
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.decomposition import PCA

# 配置字体为中文
rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
rcParams['axes.unicode_minus'] = False   # 正常显示负号

# 加载鸢尾花数据集
iris = load_iris()
data = pd.DataFrame(iris.data, columns=['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'])
data['类别'] = iris.target

# 将类别数字替换为中文名称
data['类别'] = data['类别'].replace({0: '山鸢尾', 1: '变色鸢尾', 2: '维吉尼亚鸢尾'})

# 绘制特征两两关系图
sns.pairplot(data, hue='类别', diag_kind='kde',
             plot_kws={'alpha': 0.7},
             diag_kws={'fill': True})  # 更新为fill=True

# 设置图表标题
plt.suptitle("鸢尾花数据集特征两两关系图", y=1.02, fontsize=16)
plt.show()

# 计算相关性矩阵
correlation_matrix = data[['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']].corr()

# 绘制热力图
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("特征相关性热力图")
plt.show()

# 进行PCA降维
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data[['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']])

# 使用数值类别代替中文类别进行颜色标注
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=iris.target, cmap='viridis')  # 使用iris.target
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.title('PCA降维后的鸢尾花数据分布')
plt.colorbar(label='类别')  # 显示类别的颜色条
plt.show()
# 绘制小提琴图
sns.violinplot(x='类别', y='花萼长度', data=data)
plt.title('不同类别鸢尾花花萼长度分布')
plt.show()