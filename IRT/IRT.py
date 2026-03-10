import pandas as pd
import json
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 第一部分：数据处理与 1PL(Rasch) 模型训练
# ==========================================

print("开始读取和处理数据...")
# 1. 读取 jsonl 数据
data = []
with open("train_dataset.filtered.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line.strip()))

df = pd.DataFrame(data)

# 2. 数据预处理
# 将 "IsCorrect" 字段的 正确/错误 转换为 1 和 0
df['score'] = df['IsCorrect'].map({'正确': 1, '错误': 0})

# 给用户ID和题目ID加上前缀，确保转为字符串类别，以防混淆
df['UserId_str'] = 'U_' + df['UserId'].astype(str)
df['question_id_str'] = 'Q_' + df['question_id'].astype(str)

# 3. 构建独热编码（One-Hot）的特征矩阵
# 使用稀疏矩阵优化内存和运算速度
print("构建特征矩阵...")
X_users = pd.get_dummies(df['UserId_str'])
X_items = pd.get_dummies(df['question_id_str'])

X_users_sp = sp.csr_matrix(X_users.values)
X_items_sp = sp.csr_matrix(X_items.values)
# 横向拼接用户特征和题目特征
X = sp.hstack([X_users_sp, X_items_sp])
y = df['score'].values

# 4. 训练逻辑回归模型估计 IRT 参数
# fit_intercept=False 确保严格按照 Rasch 模型的公式
# C=1.0 起到 L2 正则化的作用（充当贝叶斯先验），防止由于某些题目没人答对导致参数发散
print("开始训练模型...")
model = LogisticRegression(fit_intercept=False, C=1.0, solver='lbfgs', max_iter=2000)
model.fit(X, y)

# 5. 提取能力值 \theta 和 难度值 b
user_cols = X_users.columns
item_cols = X_items.columns

# 用户部分的参数 = \theta
theta_estimates = model.coef_[0][:len(user_cols)]
# 题目部分的参数 = -b, 所以难度 b = -系数
b_estimates = -model.coef_[0][len(user_cols):]

# 6. 保存和展示结果
user_df = pd.DataFrame({
    'UserId': [c.replace('U_', '') for c in user_cols], 
    'theta_ability': theta_estimates
})

item_df = pd.DataFrame({
    'question_id': [c.replace('Q_', '') for c in item_cols], 
    'b_difficulty': b_estimates
})

# 按能力值和难度排序
user_df = user_df.sort_values(by='theta_ability', ascending=False)
item_df = item_df.sort_values(by='b_difficulty', ascending=False)

# 导出为 CSV 文件
user_df.to_csv('user_ability.csv', index=False)
item_df.to_csv('item_difficulty.csv', index=False)

print("\n被试者能力水平前5名：")
print(user_df.head(5))

print("\n题目难度前5名：")
print(item_df.head(5))


# ==========================================
# 第二部分：结果可视化
# ==========================================
print("\n开始生成可视化图表...")

# 设置绘图风格
sns.set(style="whitegrid")
# 尽量支持中文显示，防止负号显示为方块
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  
plt.rcParams['axes.unicode_minus'] = False

# --- 图1：可视化被试者能力分布 ---
plt.figure(figsize=(10, 6))
sns.histplot(user_df['theta_ability'], bins=10, kde=True, color='skyblue')
plt.title('Distribution of User Ability (Theta)', fontsize=15)
plt.xlabel('Ability Level (Theta)', fontsize=12)
plt.ylabel('Count of Users', fontsize=12)
plt.tight_layout()
plt.savefig('user_ability_dist.png')
plt.show()

# --- 图2：可视化题目难度分布 ---
plt.figure(figsize=(10, 6))
sns.histplot(item_df['b_difficulty'], bins=30, kde=True, color='salmon')
plt.title('Distribution of Item Difficulty (b)', fontsize=15)
plt.xlabel('Difficulty Level (b)', fontsize=12)
plt.ylabel('Count of Items', fontsize=12)
plt.tight_layout()
plt.savefig('item_difficulty_dist.png')
plt.show()

# --- 图3：可视化项目特征曲线 (ICC) ---
# 选择几道具有代表性的题目（最易、较易、中等、较难、最难）
item_df_sorted = item_df.sort_values('b_difficulty').reset_index(drop=True)
num_items = len(item_df_sorted)

sample_indices = [0, num_items//4, num_items//2, 3*num_items//4, num_items-1]
sample_items = item_df_sorted.iloc[sample_indices]

# 定义计算ICC答对概率的函数 (单参数 1PL / Rasch 模型)
def icc_prob(theta, b):
    return 1 / (1 + np.exp(-(theta - b)))

# 生成一系列的能力值 theta（通常取 -4 到 4 之间）
theta_range = np.linspace(-4, 4, 100)

plt.figure(figsize=(12, 8))
colors = sns.color_palette("husl", len(sample_items))

# 循环绘制每一道选中题目的ICC曲线
for i, (idx, row) in enumerate(sample_items.iterrows()):
    b = row['b_difficulty']
    q_id = row['question_id']
    probs = icc_prob(theta_range, b)
    plt.plot(theta_range, probs, label=f'Question {q_id} (b={b:.2f})', color=colors[i], linewidth=2.5)

# 图表装饰与标签
plt.title('Item Characteristic Curves (ICC) for Selected Questions', fontsize=16)
plt.xlabel('User Ability Level (Theta)', fontsize=14)
plt.ylabel('Probability of Correct Answer P(X=1)', fontsize=14)
plt.axvline(0, color='gray', linestyle='--', alpha=0.5) # 辅助线：平均能力 0
plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5) # 辅助线：50% 答对概率
plt.legend(title='Items', loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('icc_plot.png')
plt.show()

print("程序运行结束，图表已保存。")