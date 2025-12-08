import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error
import joblib
from time import time

# ---------- 数据读取 ----------
def read_data(file_path):
    user_item_data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            user_id = int(parts[0])
            item_ids = list(map(int, parts[1:]))
            for item_id in item_ids:
                user_item_data.append([user_id, item_id])
    return pd.DataFrame(user_item_data, columns=['user_id', 'item_id'])

click_data = read_data('pv.txt')
cart_data = read_data('cart.txt')
train_data = read_data('train.txt')

click_data['treatment'] = 0  # 点击行为 T=0
cart_data['treatment'] = 1   # 加购行为 T=1

all_data = pd.concat([click_data, cart_data], ignore_index=True)
purchased_items = set(map(tuple, train_data.values))
all_data['outcome'] = all_data.apply(lambda row: 1 if (row['user_id'], row['item_id']) in purchased_items else 0, axis=1)

# ---------- 构造用户特征 ----------
user_features = all_data.groupby('user_id').agg(
    total_clicks=('treatment', lambda x: (x == 0).sum()),
    total_adds_to_cart=('treatment', lambda x: (x == 1).sum()),
    total_interactions=('item_id', 'count')
).reset_index()

all_data = pd.merge(all_data, user_features, on='user_id', how='left')

X = all_data[['total_clicks', 'total_adds_to_cart', 'total_interactions']]
T = all_data['treatment']
Y = all_data['outcome']

# ---------- 拆分训练集/测试集 ----------
X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(X, T, Y, test_size=0.2, random_state=42)

# ---------- 训练 e(X) 倾向得分 ----------
model_e = GradientBoostingClassifier()
model_e.fit(X_train, T_train)
e_X = model_e.predict_proba(X_test)[:, 1]  # e(X) = P(T=1|X)

# ---------- 训练 m0(X) 和 m1(X) ----------
model_m0 = GradientBoostingRegressor()
model_m1 = GradientBoostingRegressor()

# 注意：m0 用 T=0 的数据，m1 用 T=1 的数据
model_m0.fit(X_train[T_train == 0], Y_train[T_train == 0])
model_m1.fit(X_train[T_train == 1], Y_train[T_train == 1])

m0_X = model_m0.predict(X_test)
m1_X = model_m1.predict(X_test)

# ---------- m_T(X_i)（按 T 选 m0 还是 m1） ----------
T_test_array = T_test.values
Y_test_array = Y_test.values
m_T_X = np.where(T_test_array == 0, m0_X, m1_X)

# ---------- 计算 φ_i ----------
phi = m1_X - m0_X + ((T_test_array - e_X) / (e_X * (1 - e_X))) * (Y_test_array - m_T_X)

# ---------- 保存结果 ----------
user_ids = all_data.iloc[X_test.index]['user_id'].reset_index(drop=True)
df_result = pd.DataFrame({
    'user_id': user_ids,
    'phi': phi,
    'm0': m0_X,
    'm1': m1_X,
    'e(X)': e_X
})
df_result.to_csv('phi_result.csv', index=False)
print("已保存 φ_i 结果到 phi_result.csv")
