# import numpy as np
# import pickle
# from scipy.sparse import csr_matrix
# import scipy.sparse as sp
# import random




# def get_random_subset_by_percentage(original_set, percentage):
#     # 计算要选取的元素数量
#     sample_size = max(0, int(len(original_set) * percentage))
    
#     # 从原集合中随机选取指定数量的元素，并将结果转换为集合返回
#     return set(random.sample(original_set, sample_size))


# def get_aug(dataset):
#     user_interactions = {i: {} for i in range(len(behs))}
#     for i in range(len(behs)):
#         with open(f"dataset/{dataset}/{behs[i]}.txt", 'r') as f:
#             for line in f:
#                 parts = line.strip().split()
#                 user_id = parts[0]
#                 item_ids = set(parts[1:])
#                 if user_id in user_interactions[i]:
#                     user_interactions[i][user_id].update(item_ids)
#                 else:
#                     user_interactions[i][user_id] = item_ids
#     augmented_train = {}
     
#     for user_id in user_interactions[len(behs)-1].keys():
#         aux_items = set()
#         for i in range(len(behs) - 1):
#             items = user_interactions[i].get(user_id, set())
#             aux_items.update(items)
#         train_items = user_interactions[len(behs)-1].get(user_id)
        
#         # 更新 train 数据
#         random_subset = get_random_subset_by_percentage(aux_items, 0.5)
#         augmented_train[user_id] = train_items | random_subset

#     # 保存增强后的数据到 aug.txt
#     with open(f'dataset/{dataset}/aug.txt', 'w') as f:
#         for user_id, item_ids in augmented_train.items():
#             f.write(f"{user_id} {' '.join(item_ids)}\n")

#     print("增强后的文件已保存为 aug.txt")


# def add_unique_highest_scores_from_multiple_matrices(matrices, ratio=0.2):
#     """
#     从多个辅助矩阵中选择目标矩阵中不存在的交互项，并将它们添加到目标矩阵中。
    
#     参数:
#     matrices: list of csr_matrix
#         CSR矩阵的列表，最后一个矩阵是目标矩阵，其他矩阵用于选择高分交互。
#     ratio: float
#         从目标矩阵中现有交互项数量的比例，值介于 0 和 1 之间。
    
#     返回:
#     csr_matrix: 更新后的最后一个矩阵，包含所选的高分交互。
#     """
    
#     # 获取最后一个矩阵（目标矩阵）
#     last_matrix = matrices[-1]
    
#     # 获取目标矩阵中已有的交互项
#     existing_rows, existing_cols = last_matrix.nonzero()
#     existing_interactions = set(zip(existing_rows, existing_cols))  # 使用集合加快查找速度
    
#     # 计算需要添加的交互项数量
#     total_existing_interactions = len(existing_rows)
#     total_to_select = int(total_existing_interactions * ratio)
    
#     # 初始化用于存储所有选定的交互项
#     all_rows, all_cols, all_scores = [], [], []
    
#     # 遍历每个辅助矩阵，选择交互项
#     for matrix in matrices[:-1]:
#         # 收集非零项
#         row, col = matrix.nonzero()
#         score = matrix.data
        
#         # 按分数降序排序
#         sorted_idx = np.argsort(-score)
        
#         # 初始化计数器
#         selected_count = 0
        
#         # 遍历排序后的索引，选择目标矩阵中没有的交互项
#         for idx in sorted_idx:
#             if selected_count >= total_to_select:
#                 break
            
#             current_row = row[idx]
#             current_col = col[idx]
#             current_interaction = (current_row, current_col)
            
#             # 检查该交互项是否已存在于目标矩阵中
#             if current_interaction not in existing_interactions:
#                 all_rows.append(current_row)
#                 all_cols.append(current_col)
#                 all_scores.append(score[idx])
#                 selected_count += 1
    
#     # 将所有选定的交互项展平
#     all_rows = np.array(all_rows)
#     all_cols = np.array(all_cols)
#     all_scores = np.array(all_scores)
    
#     # 创建一个新矩阵，包含所有选定的高分交互
#     shape = last_matrix.shape
#     new_matrix = csr_matrix((all_scores, (all_rows, all_cols)), shape=shape)
    
#     # 将新矩阵与最后一个矩阵相加
#     updated_last_matrix = last_matrix + new_matrix
    
#     return updated_last_matrix


# dataset_list = ["Beibei", "Taobao", "IJCAI", "Tmall"]
# # for dataset in dataset_list:
# # # 数据集参数设置
# # # 文件列表，包含多个txt文件
# #     if dataset == "Beibei":
# #         behs = ['pv', 'cart', 'train']
# #         beh_rat = [1, 2, 3]
# #         n_user = 21716
# #         n_item = 7977
# #     elif dataset == "Taobao":
# #         behs = ['pv', 'cart', 'train']
# #         beh_rat = [1, 2, 3]
# #         n_user = 48749
# #         n_item = 39493
# #     elif dataset == "IJCAI":
# #         beh_rat = [1, 2, 2, 3]
# #         behs = ['click', 'fav', 'cart', 'train']
# #         n_user = 17435
# #         n_item = 35920
# #     elif dataset == "Tmall":
# #         behs = ['pv', 'fav', 'cart', 'train']
# #         beh_rat = [1, 2, 2, 3]
# #         n_user = 31882
# #         n_item = 31232
# dataset = "IJCAI"
# if dataset == "Beibei":
#     behs = ['pv', 'cart', 'train']
#     beh_rat = [1, 2, 3]
#     n_user = 21716
#     n_item = 7977
# elif dataset == "Taobao":
#     behs = ['pv', 'cart', 'train']
#     beh_rat = [1, 2, 3]
#     n_user = 48749
#     n_item = 39493
# elif dataset == "IJCAI":
#     beh_rat = [1, 2, 2, 3]
#     behs = ['click', 'fav', 'cart', 'train']
#     n_user = 17435
#     n_item = 35920
# elif dataset == "Tmall":
#     behs = ['pv', 'fav', 'cart', 'train']
#     beh_rat = [1, 2, 2, 3]
#     n_user = 31882
#     n_item = 31232
# print(f"{dataset}")
# print(f"{dataset}")
# get_aug(dataset)
#     # with open(f"denoised/denoised_{dataset}-0.5.pkl", 'rb') as file:
#     #         user_item_matrix = pickle.load(file)
#     # with open(f"denoised/{dataset}-0.5.pkl", 'rb') as file:
#     #     user_item_matrix_buy = pickle.load(file)
#     # user_item_matrix.append(user_item_matrix_buy[-1])
#     # updated_last_matrix = add_unique_highest_scores_from_multiple_matrices(user_item_matrix, ratio=0.2)  # 从每个辅助矩阵中加入目标矩阵现有交互项的20%
#     # sp.save_npz(f"denoised/{dataset}-aug.npz", updated_last_matrix)
#     # print("Done")
    
import numpy as np
import pickle
from scipy.sparse import csr_matrix
import scipy.sparse as sp


def add_unique_highest_scores_from_multiple_matrices(matrices, ratio):
    """
    从多个辅助矩阵中选择目标矩阵中不存在的交互项，并将它们添加到目标矩阵中。
    
    参数:
    matrices: list of csr_matrix
        CSR矩阵的列表，最后一个矩阵是目标矩阵，其他矩阵用于选择高分交互。
    ratio: float
        从目标矩阵中现有交互项数量的比例，值介于 0 和 1 之间。
    
    返回:
    csr_matrix: 更新后的最后一个矩阵，包含所选的高分交互。
    """
    
    # 获取最后一个矩阵（目标矩阵）
    last_matrix = matrices[-1].copy()
    
    # 获取目标矩阵中已有的交互项
    existing_rows, existing_cols = last_matrix.nonzero()
    existing_interactions = set(zip(existing_rows, existing_cols))  # 使用集合加快查找速度
    
    # 计算需要添加的交互项数量
    total_existing_interactions = len(existing_rows)
    total_to_select = int(total_existing_interactions * ratio)
    
    # 初始化用于存储所有选定的交互项
    all_rows, all_cols, all_scores = [], [], []
    
    # 遍历每个辅助矩阵，选择交互项
    for matrix in matrices[:-1]:
        # 收集非零项
        row, col = matrix.nonzero()
        score = matrix.data
        
        # 按分数降序排序
        sorted_idx = np.argsort(-score)
        
        # 初始化计数器
        selected_count = 0
        
        # 遍历排序后的索引，选择目标矩阵中没有的交互项
        for idx in sorted_idx:
            if selected_count >= total_to_select:
                break
            
            current_row = row[idx]
            current_col = col[idx]
            current_interaction = (current_row, current_col)
            
            # 检查该交互项是否已存在于目标矩阵中
            if current_interaction not in existing_interactions:
                all_rows.append(current_row)
                all_cols.append(current_col)
                all_scores.append(score[idx])
                selected_count += 1
    
    # 将所有选定的交互项展平
    all_rows = np.array(all_rows)
    all_cols = np.array(all_cols)
    all_scores = np.array(all_scores)
    
    # 创建一个新矩阵，包含所有选定的高分交互
    shape = last_matrix.shape
    new_matrix = csr_matrix((all_scores, (all_rows, all_cols)), shape=shape)
    
    # 将新矩阵与最后一个矩阵相加
    updated_last_matrix = last_matrix + new_matrix
    
    return updated_last_matrix


dataset_list = ["Beibei", "Taobao", "IJCAI", "Tmall"]
for dataset in dataset_list:
# 数据集参数设置
# 文件列表，包含多个txt文件
    if dataset == "Beibei":
        behs = ['pv', 'cart', 'train']
        beh_rat = [1, 2, 3]
        n_user = 21716
        n_item = 7977
    elif dataset == "Taobao":
        behs = ['pv', 'cart', 'train']
        beh_rat = [1, 2, 3]
        n_user = 48749
        n_item = 39493
    elif dataset == "IJCAI":
        beh_rat = [1, 2, 2, 3]
        behs = ['click', 'fav', 'cart', 'train']
        n_user = 17435
        n_item = 35920
    elif dataset == "Tmall":
        behs = ['pv', 'fav', 'cart', 'train']
        beh_rat = [1, 2, 2, 3]
        n_user = 31882
        n_item = 31232
    print(f"{dataset}")
    with open(f"denoised/denoised_{dataset}-low-0.1.pkl", 'rb') as file:
            user_item_matrix = pickle.load(file)
    ratio_list = [0.1, 0.3, 0.4, 0.5]
    for ratio in ratio_list:
        updated_last_matrix = add_unique_highest_scores_from_multiple_matrices(user_item_matrix, ratio)  # 从每个辅助矩阵中加入目标矩阵现有交互项的20%
        sp.save_npz(f"denoised/{dataset}-aug{ratio}.npz", updated_last_matrix)
        print("Done")
