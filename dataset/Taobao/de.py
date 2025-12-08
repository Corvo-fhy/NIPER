import pandas as pd

# 读取 CSV
df = pd.read_csv('treatment_effect_result.csv')

# 根据 user_id 去重（默认保留第一次出现的记录）
df_unique = df.drop_duplicates(subset='user_id')

# 可选：保存去重后的结果
df_unique.to_csv('treatment_effect_unique.csv', index=False)

print(f"原始数据共 {len(df)} 条，去重后剩余 {len(df_unique)} 条")
print("已保存去重结果至 treatment_effect_unique.csv")
