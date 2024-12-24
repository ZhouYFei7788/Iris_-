import cx_Oracle
import pandas as pd
import matplotlib.pyplot as plt

# 数据库连接信息
dsn = cx_Oracle.makedsn("localhost", 1521, service_name="XE")  # 替换为你的数据库信息
connection = cx_Oracle.connect(user="C##YUANWEIHUA", password="root", dsn=dsn)

# 创建数据表
create_table_sql = """
CREATE TABLE "模型训练日志" (
    "编号" NUMBER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,  -- 自增ID
    "轮次" NUMBER NOT NULL,                                    -- 训练轮次
    "模型名称" VARCHAR2(50) NOT NULL,                         -- 模型名称
    "交叉验证分组" NUMBER,                                     -- 交叉验证的分组编号
    "训练集准确率" FLOAT,                                      -- 训练集的准确率
    "验证集准确率" FLOAT,                                      -- 验证集的准确率
    "训练集损失值" FLOAT,                                      -- 训练集的损失值
    "验证集损失值" FLOAT,                                      -- 验证集的损失值
    "精确率" FLOAT,                                           -- 精确率
    "召回率" FLOAT,                                           -- 召回率
    "F1值" FLOAT,                                             -- F1值
    "混淆矩阵" CLOB,                                           -- 混淆矩阵（可以存储为 CLOB 类型）
    "AUC" FLOAT,                                              -- AUC值
    "创建时间" TIMESTAMP DEFAULT CURRENT_TIMESTAMP            -- 创建时间
)
"""
try:
    with connection.cursor() as cursor:
        cursor.execute(create_table_sql)
    print("表创建成功！")
except cx_Oracle.DatabaseError as e:
    print("表已存在或其他错误：", e)

# 插入模拟数据
insert_sql = """
INSERT INTO "模型训练日志" ("轮次", "模型名称", "交叉验证分组", "训练集准确率", "验证集准确率", 
"训练集损失值", "验证集损失值", "精确率", "召回率", "F1值", "混淆矩阵", "AUC")
VALUES (:轮次, :模型名称, :交叉验证分组, :训练集准确率, :验证集准确率, 
:训练集损失值, :验证集损失值, :精确率, :召回率, :F1值, :混淆矩阵, :AUC)
"""

# 模拟模型训练结果
data = [
    (1, "随机森林", 1, 0.95, 0.92, 0.05, 0.08, 0.93, 0.91, 0.92, '{"TP":50,"FP":5}', 0.98),
    (2, "逻辑回归", 2, 0.89, 0.87, 0.11, 0.13, 0.88, 0.86, 0.87, '{"TP":45,"FP":8}', 0.95)
]

with connection.cursor() as cursor:
    for row in data:
        cursor.execute(insert_sql, {
            "轮次": row[0],
            "模型名称": row[1],
            "交叉验证分组": row[2],
            "训练集准确率": row[3],
            "验证集准确率": row[4],
            "训练集损失值": row[5],
            "验证集损失值": row[6],
            "精确率": row[7],
            "召回率": row[8],
            "F1值": row[9],
            "混淆矩阵": row[10],
            "AUC": row[11]
        })
connection.commit()
print("数据插入成功！")

# 从数据库读取数据
read_sql = "SELECT * FROM \"模型训练日志\""
df = pd.read_sql(read_sql, con=connection)

# 展示数据
print(df)

# 可视化 AUC 值
plt.figure(figsize=(10, 6))
plt.bar(df["模型名称"], df["AUC"], color='skyblue')
plt.title("模型 AUC 值对比")
plt.xlabel("模型名称")
plt.ylabel("AUC 值")
plt.show()

connection.close()
