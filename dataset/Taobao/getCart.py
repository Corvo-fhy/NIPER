import pandas as pd
import numpy as np

# 读取 treatment_effect_result.csv
df = pd.read_csv('treatment_effect_unique.csv')

# 计算权重 w_CART 和 w_CTRL
df['w_cart'] = np.exp(df['treatment_effect']) / (np.exp(df['treatment_effect']) + 1)
df['w_click'] = 1 - df['w_cart']

# 保存结果
df.to_csv('treatment_effect_with_weights.csv', index=False)
print("已保存添加权重的结果至 treatment_effect_with_weights.csv")
