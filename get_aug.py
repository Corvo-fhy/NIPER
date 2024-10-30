import numpy as np
import pickle
from scipy.sparse import csr_matrix
import scipy.sparse as sp


def add_unique_highest_scores_from_multiple_matrices(matrices, ratio):

    

    last_matrix = matrices[-1].copy()
    

    existing_rows, existing_cols = last_matrix.nonzero()
    existing_interactions = set(zip(existing_rows, existing_cols))  # 使用集合加快查找速度

    total_existing_interactions = len(existing_rows)
    total_to_select = int(total_existing_interactions * ratio)
    

    all_rows, all_cols, all_scores = [], [], []
    

    for matrix in matrices[:-1]:

        row, col = matrix.nonzero()
        score = matrix.data
        

        sorted_idx = np.argsort(-score)
        

        selected_count = 0
        

        for idx in sorted_idx:
            if selected_count >= total_to_select:
                break
            
            current_row = row[idx]
            current_col = col[idx]
            current_interaction = (current_row, current_col)
            
            
            if current_interaction not in existing_interactions:
                all_rows.append(current_row)
                all_cols.append(current_col)
                all_scores.append(score[idx])
                selected_count += 1
    
    
    all_rows = np.array(all_rows)
    all_cols = np.array(all_cols)
    all_scores = np.array(all_scores)
    
    
    shape = last_matrix.shape
    new_matrix = csr_matrix((all_scores, (all_rows, all_cols)), shape=shape)
    
    
    updated_last_matrix = last_matrix + new_matrix
    
    return updated_last_matrix


dataset_list = ["Beibei", "Taobao", "IJCAI", "Tmall"]
for dataset in dataset_list:

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
