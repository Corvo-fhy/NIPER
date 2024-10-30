import numpy as np
import scipy.sparse as sp
import os
import pickle
from tqdm import tqdm



def load_txt_file(file_path):
    user_item_matrix = sp.lil_matrix((n_user, n_item))

    with open(file_path, 'r') as f:
        for line in f:
            line_data = list(map(int, line.strip().split()))
            for i in line_data[1:]:
                user_item_matrix[line_data[0], i] = 1
            
    return user_item_matrix


def build_matrices(user_item_matrix):
    num_users = user_item_matrix.shape[0]
    num_items = user_item_matrix.shape[1]

    user_matrix = sp.lil_matrix((num_users, num_users))
    item_matrix = sp.lil_matrix((num_items, num_items))
    
    user_item_matrix = user_item_matrix.tocsr()
    item_user_matrix = user_item_matrix.T


    for user_a in tqdm(range(num_users),desc="process user"):
        vector = user_item_matrix[user_a].todense().A1
        result  = user_item_matrix.dot(vector)
        for i in range(num_users):
            if result[i] != 0:
                user_matrix[user_a, i] = 1
                user_matrix[i, user_a] = 1
 

    for item_a in tqdm(range(num_items), desc="process item"):
        vector = item_user_matrix[item_a].todense().A1
        result  = item_user_matrix.dot(vector)
        for i in range(num_items):
            if result[i] != 0:
                item_matrix[item_a, i] = 1
                item_matrix[i, item_a] = 1

    return user_matrix.tocsr(), item_matrix.tocsr()

def union_matrices(matrix_list):
    result = matrix_list[0].copy()  
    for matrix in matrix_list[1:]:
        result = result + matrix  
    

    result[result > 0] = 1
    return result


def process_all_files(n_user, n_item):
    user_matrices = []
    item_matrices = []

    with open(f"denoised/denoised_{dataset}-0.5.pkl", 'rb') as file:
        user_item_matrix = pickle.load(file)

    with open(f"denoised/{dataset}-0.5.pkl", 'rb') as file:
        user_item_matrix_buy = pickle.load(file)
    user_item_matrix.append(user_item_matrix_buy[-1])

    for i in range(len(behs)):
        # user_item_matrix = load_txt_file(file)
        
        user_matrix, item_matrix = build_matrices(user_item_matrix[i])
        
        user_matrices.append(user_matrix)
        item_matrices.append(item_matrix)

    
    final_user_matrix = union_matrices(user_matrices)
    
    final_item_matrix = union_matrices(item_matrices)


    sp.save_npz(f"Sim/{dataset}_user_matrix.npz", final_user_matrix)
    sp.save_npz(f"Sim/{dataset}_item_matrix.npz", final_item_matrix)
    print(f"{dataset}:Done")


def denoise_matrix():
    with open(f"denoised/denoised_{dataset}-low-0.5.pkl", 'rb') as file:
        user_item_matrix = pickle.load(file)
    # user_item_matrix = sp.load_npz(f"denoised/{dataset}-aug.npz")
    user_matrix, item_matrix = build_matrices(user_item_matrix[-1])
    
    sp.save_npz(f"Sim/{dataset}_user_matrix.npz", user_matrix)
    sp.save_npz(f"Sim/{dataset}_item_matrix.npz", item_matrix)
    print(f"{dataset}:Done")


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
    
    # process_all_files(n_user, n_item)
    denoise_matrix()
