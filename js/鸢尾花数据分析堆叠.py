import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from matplotlib import rcParams
###面积图（Area Plot）是一种堆叠的图表，用于显示不同类别的数据的数量随时间的变化情况。###

# 配置字体为中文
rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
rcParams['axes.unicode_minus'] = False   # 正常显示负号

# 加载鸢尾花数据集
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)

# 为了演示，将数据归一化
normalized_data = data / data.max()

# 绘制面积图
normalized_data.plot.area(alpha=0.6, figsize=(10, 6))
plt.title("鸢尾花数据集面积图")  # 图表标题
plt.ylabel("规范化值")            # Y轴标签
plt.xlabel("样本索引")           # X轴标签
plt.legend(loc="upper right", labels=["花萼长度", "花萼宽度", "花瓣长度", "花瓣宽度"])  # 图例
plt.show()
