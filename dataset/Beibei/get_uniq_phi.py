import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('phi_result.csv')

# 去重：每个 user_id 只保留第一行
df_unique = df.drop_duplicates(subset='user_id')

# 计算 φ 的总体平均
phi_mean = df_unique['phi'].mean()

print(f"所有唯一用户的平均 φ 值为：{phi_mean:.6f}")
