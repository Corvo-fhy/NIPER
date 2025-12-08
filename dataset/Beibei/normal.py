import pandas as pd

# 读取之前保存的含权重的CSV
df = pd.read_csv('treatment_effect_with_weights.csv')

# 对 w_cart 做标准化
cart_mean = df['w_cart'].mean()
cart_std = df['w_cart'].std()
df['w_cart_normalized'] = (df['w_cart'] - cart_mean) / cart_std

# 对 w_click 做标准化
click_mean = df['w_click'].mean()
click_std = df['w_click'].std()
df['w_click_normalized'] = (df['w_click'] - click_mean) / click_std

# 保存归一化后的结果
df.to_csv('treatment_effect_with_normalized_weights.csv', index=False)
print("已保存归一化后的权重到 treatment_effect_with_normalized_weights.csv")
print("click_mean:", click_mean)
print("cart_mean:", cart_mean)