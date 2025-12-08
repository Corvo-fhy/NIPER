import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from econml.dml import CausalForestDML
from sklearn.metrics import mean_squared_error
import joblib
from time import time


# 假设文件路径
click_file = 'pv.txt'
cart_file = 'cart.txt'
train_file = 'train.txt'

# 读取行为数据的函数
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

# 读取数据
click_data = read_data(click_file)
cart_data = read_data(cart_file)
train_data = read_data(train_file)

# 生成干预变量 T 和目标变量 Y
click_data['treatment'] = 1  # 点击行为对应 T=1
cart_data['treatment'] = 2   # 加入购物车行为对应 T=2

# 合并所有的行为数据
all_data = pd.concat([click_data, cart_data], ignore_index=True)

# 为每个用户生成购买行为的目标变量
purchased_items = set(map(tuple, train_data.values))  # 用户购买的物品列表
all_data['outcome'] = all_data.apply(lambda row: 1 if (row['user_id'], row['item_id']) in purchased_items else 0, axis=1)

# 生成用户特征，统计每个用户的点击、加入购物车行为
user_features = all_data.groupby('user_id').agg(
    total_clicks=('treatment', lambda x: (x == 1).sum()),
    total_adds_to_cart=('treatment', lambda x: (x == 2).sum()),
    total_interactions=('item_id', 'count')  # 用户交互的物品数量
).reset_index()

# 合并数据以便进行训练
all_data = pd.merge(all_data, user_features, on='user_id', how='left')

# 特征矩阵 X（用户特征），干预变量 T（点击或加购），目标变量 Y（是否购买）
X = all_data[['total_clicks', 'total_adds_to_cart', 'total_interactions']]
T = all_data['treatment']
Y = all_data['outcome']

# 将数据分为训练集和测试集
X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(X, T, Y, test_size=0.2, random_state=42)

# 训练因果森林模型
causal_forest = CausalForestDML(n_estimators=100, min_samples_leaf=10, random_state=42)

print("开始训练 CausalForestDML 模型...")
start_time = time()

causal_forest.fit(Y_train, T_train, X=X_train)

print(f"训练完成，耗时 {time() - start_time:.2f} 秒")

# 预测每个用户的个性化处理效应
treatment_effect = causal_forest.effect(X_test)

# 输出个性化处理效应
print("个性化处理效应预测结果：", treatment_effect)

# 获取测试集对应的 user_id（注意保持顺序）
user_ids = all_data.iloc[X_test.index]['user_id'].reset_index(drop=True)

# 构建 DataFrame
df_result = pd.DataFrame({
    'user_id': user_ids,
    'treatment_effect': treatment_effect
})

# 保存为 CSV 文件
df_result.to_csv('treatment_effect_result.csv', index=False)
print("已保存个性化处理效应结果至 treatment_effect_result.csv")


# 计算预测效果（如均方误差）
# Y_pred = causal_forest.predict(X_test)
# mse = mean_squared_error(Y_test, Y_pred)

# print(f"模型的均方误差 (MSE)：{mse}")


# 保存模型
joblib.dump(causal_forest, 'causal_forest_model.pkl')
print("模型已保存为 causal_forest_model.pkl")