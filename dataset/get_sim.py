import os
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import pickle

def read_data(filepath):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            user_id = int(parts[0])
            item_ids = [int(item) for item in parts[1:]]
            data.append((user_id, item_ids))
    return data


def find_common_interactions(data):
    item_user_map = defaultdict(set)
    for user_id, item_ids in data:
        for item_id in item_ids:
            item_user_map[item_id].add(user_id)

    common_users = {item: sorted(users) for item, users in item_user_map.items() if len(users) > 1}
    user_indices = np.zeros((n_user, n_user), dtype=bool)
    for users in tqdm(common_users.values(), desc="Processing users"):
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                x, y = users[i], users[j]
                user_indices[x, y] = 1
                user_indices[y, x] = 1  # 对称的坐标也标记为1
    return user_indices


def save_common_users(common_users, output_filepath):
    with open(output_filepath, 'w') as f:
        for item in sorted(common_users.keys()):
            users = common_users[item]
            f.write(f"{item} {' '.join(map(str, users))}\n")


def process_specific_files(directory, filenames):
    for root, _, files in os.walk(directory):
        for filename in filenames:
            if filename in files:
                filepath = os.path.join(root, filename)
                print(f"Processing {filepath}")
                data = read_data(filepath)
                common_users = find_common_interactions(data)
                common_items = find_common_irems(data)
                with open(f'{filename}_user_indices.pkl', 'wb') as f:
                    pickle.dump(common_users, f)
                with open(f'{filename}_item_indices.pkl', 'wb') as f:
                    pickle.dump(common_items, f)
                print(f"Saved")


def find_common_irems(data):
    item_dict = {}
    for user_id, item_ids in data:
        item_dict[user_id] = item_ids
    item_indices = np.zeros((n_item, n_item), dtype=bool)
    for users in tqdm(item_dict.values(), desc="Processing items"):
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                x, y = users[i], users[j]
                item_indices[x, y] = 1
                item_indices[y, x] = 1  # 对称的坐标也标记为1
    return item_indices

# 设置数据集文件夹路径和要处理的文件名
dataset_directory = 'Tmall'
n_user = 21716
n_item = 7977

filenames_to_process = ['pv.txt', 'cart.txt','fav.txt', 'train.txt']  # 可以添加更多文件名

# 处理指定的文件
process_specific_files(dataset_directory, filenames_to_process)


# For beibei
# n_user = 21716
# n_item = 7977


# For Tmall
# n_user = 31882
# n_item = 31230