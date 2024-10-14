import os
import numpy as np
import pickle
import time
import scipy.sparse as sp
from tqdm import tqdm



def save_sparse_matrices(matrix_list, base_filename='sparse_matrix'):
    sparse_matrices = [sp.csr_matrix(arr) for arr in matrix_list]
    for idx, sparse_matrix in enumerate(sparse_matrices):
        filename = f'csr/{dataset}/{base_filename}_{idx + 1}.npz'
        sp.save_npz(filename, sparse_matrix)
        print(f'Saved: {filename}')


def load_sparse_matrices(count, base_filename='sparse_matrix'):
    loaded_sparse_matrices = []
    for idx in range(count):
        filename = f'{base_filename}_{idx + 1}.npz'
        sparse_matrix = sp.load_npz(filename)
        loaded_sparse_matrices.append(sparse_matrix)
        print(f'Loaded: {filename}')
    return loaded_sparse_matrices



def inter_dataset(behs, data_directory, num_users, num_items, beh_rat):
    # Step 1: Initialize a DOK sparse matrix
    user_item_matrix = sp.dok_matrix((num_users, num_items), dtype=np.float32)

    # Step 2: Populate the DOK matrix
    for i in range(len(behs)):
        with open(os.path.join(data_directory, behs[i]) + '.txt', 'r') as file:
            for line in file:
                data = list(map(int, line.split()))
                user_id = data[0]
                item_ids = data[1:]
                for item_id in item_ids:
                    # Increment the value for the specific (user_id, item_id)
                    user_item_matrix[user_id, item_id] += beh_rat[i]
    return user_item_matrix


def intra_dataset(behs, data_directory, num_users, num_items, sim_user_mat):
    # Initialize the user-item interaction matrix
    user_item_intra_matrix_list = []

    # Step 2: Populate the matrix
    for i in range(len(behs)):
        user_item_matrix = np.zeros((num_users, num_items), dtype=int)
        with open(os.path.join(data_directory, behs[i]) + '.txt', 'r') as file:
            for line in file:
                data = list(map(int, line.split()))
                user_id = data[0]
                item_ids = data[1:]
                for item_id in item_ids:
                    user_item_matrix[user_id, item_id] = 1
        start_time = time.time()
    
        for user_id in tqdm(range(num_users), desc="intra users"):
            sim_users = sim_user_mat[user_id]
            user_item_inter = user_item_matrix[user_id]
            for sim_user in sim_users.indices:
                sim_inter = user_item_matrix[sim_user]
                common_intereact = np.logical_and(user_item_inter, sim_inter)
                user_item_inter +=  common_intereact
            user_item_matrix[user_id] = user_item_inter
        print("ok")
        end_time = time.time()

        elapsed_time = end_time - start_time
        sparse_user_item_matrix = sp.csr_matrix(user_item_matrix)
        user_item_intra_matrix_list.append(sparse_user_item_matrix)

    with open(f"denoised/{dataset}_intra.pkl", 'wb') as file:
        pickle.dump(user_item_intra_matrix_list, file)    
    return user_item_intra_matrix_list


def combine(user_item_inter_matrix, user_item_intra_matrix_list, output_pre, alpha = 0.5):
    num_users = user_item_inter_matrix.shape[0]
    num_items = user_item_inter_matrix.shape[1]
    
    for i in range(len(behs)):
        user_item_matrix = np.zeros((num_users, num_items))
        with open(os.path.join(data_directory, behs[i]) + '.txt', 'r') as file:
            for line in file:
                data = list(map(int, line.split()))
                user_id = data[0]
                item_ids = data[1:]
                for item_id in item_ids:
                    user_item_matrix[user_id, item_id] = 1
        user_item_beh_matrix = user_item_inter_matrix.multiply(user_item_matrix)
        user_item_intra_matrix_list[i] = alpha * user_item_beh_matrix + (1 - alpha) * user_item_intra_matrix_list[i]
    # with open(f"{output_pre}{dataset}-{alpha}.pkl", 'wb') as file:
    #     pickle.dump(user_item_intra_matrix_list, file)        
    return user_item_intra_matrix_list

def normalize(sparse_matrix, beh_rat):
    for i in range(sparse_matrix.shape[0]):
        row_start = sparse_matrix.indptr[i]
        row_end = sparse_matrix.indptr[i+1]
        
        if row_start == row_end:
            continue
        
        row_data = sparse_matrix.data[row_start:row_end]
        
        row_min = row_data.min()
        row_max = row_data.max()

        if row_max != row_min:
            sparse_matrix.data[row_start:row_end] = 1 + (row_data - row_min) * (sum(beh_rat)-1 / (row_max - row_min))
        else:
            sparse_matrix.data[row_start:row_end] = sum(beh_rat)

    return sparse_matrix


def denoise_low_weight_elements(sparse_matrix, percentage):
    
    sparse_matrix_copy = sparse_matrix.copy()
    
    
    num_nonzero = sparse_matrix_copy.nnz
    

    num_denoise = int(num_nonzero * percentage)
    

    data = sparse_matrix_copy.data
    

    sorted_indices = np.argsort(data)
    

    selected_indices = sorted_indices[:num_denoise]
    

    sparse_matrix_copy.data[selected_indices] = 0.0
    

    sparse_matrix_copy.eliminate_zeros()

    return sparse_matrix_copy

# inter_data = sp.load_npz(f'{dataset}_inter.npz')

# with open(f"{dataset}_inter.pkl", 'rb') as file:
#     intra_data = pickle.load(file)
    
# normalize-inter
# normal_inter_data = normalize(inter_data)

# normalize-intra
# normal_intra_data_list = []
# for i in range(len(behs)):
#     normal_intra_data = normalize(intra_data[i])
#     normal_intra_data_list.append(normal_intra_data)
    
# sp.save_npz(f"{output_pre}normal_{dataset}_inter",normal_inter_data.tocsr())

# with open(f"{output_pre}normal_{dataset}_intra.pkl", 'wb') as file:
#     pickle.dump(normal_intra_data_list, file)

# inter_data = sp.load_npz(f'{output_pre}normal_{dataset}_inter.npz')

# with open(f"{output_pre}normal_{dataset}_intra.pkl", 'rb') as file:
#     intra_data = pickle.load(file)


# combine(user_item_inter_matrix=inter_data, user_item_intra_matrix_list=intra_data, output_pre=output_pre)



dataset_list = ["Beibei", "Taobao", "IJCAI", "Tmall"]
for dataset in dataset_list:
    
    data_directory = 'dataset/'
    data_directory = data_directory + dataset
    output_pre = 'denoised/'

    if dataset == "Beibei" :
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



    # with open(f"denoised/{dataset}_intra.pkl", 'rb') as file:
    #     intra_all = pickle.load(file)

    # normal_intra_data_list = []
    # for i in range(len(behs)):
    #     normal_intra_data = normalize(intra_all[i], beh_rat)
    #     normal_intra_data_list.append(normal_intra_data)

    # data1 = sp.load_npz(f'denoised/{dataset}_inter.npz')

    # alpha = 0.5
    # combined = combine(user_item_inter_matrix=data1, user_item_intra_matrix_list=normal_intra_data_list, output_pre=output_pre, alpha=alpha)
    
    # low_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    # for low in low_list:
    #     denoised_list = []
    #     for j in range(len(behs) - 1):
    #         denoised_mat = denoise_low_weight_elements(combined[j], percentage=low/beh_rat[j])
    #         denoised_list.append(denoised_mat)
            
    #     denoised_list.append(combined[-1])
    #     with open(f"{output_pre}denoised_{dataset}-low-{low}.pkl", 'wb') as file:
    #         pickle.dump(denoised_list, file)
        

    #     print(f"{dataset}-{low}: Done")
    with open(f"denoised/{dataset}_intra.pkl", 'rb') as file:
        intra_all = pickle.load(file)

    normal_intra_data_list = []
    for i in range(len(behs)):
        normal_intra_data = normalize(intra_all[i], beh_rat)
        normal_intra_data_list.append(normal_intra_data)

    data1 = sp.load_npz(f'denoised/{dataset}_inter.npz')

    alpha = 0.5
    combined = combine(user_item_inter_matrix=data1, user_item_intra_matrix_list=normal_intra_data_list, output_pre=output_pre, alpha=alpha)
    
    low_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    for low in low_list:
        denoised_list = []
        for j in range(len(behs) - 1):
            denoised_mat = denoise_low_weight_elements(combined[j], percentage=low/beh_rat[j])
            denoised_list.append(denoised_mat)
            
        denoised_list.append(combined[-1])
        with open(f"{output_pre}denoised_{dataset}-low-{low}.pkl", 'wb') as file:
            pickle.dump(denoised_list, file)
        

        print(f"{dataset}-{low}: Done")
