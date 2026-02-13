import os
import numpy as np
import pickle
import time
import scipy.sparse as sp
from tqdm import tqdm
from scipy.sparse import save_npz, load_npz



# 不同行为之间的评分不同
def inter_dataset(behs, data_directory, num_users, num_items, beh_rat):
    # Step 1: Initialize a DOK sparse matrix
    user_item_matrix = sp.dok_matrix((num_users, num_items), dtype=np.float32)

    # Step 2: Populate the DOK matrix
    for i in range(len(behs)):
        with open(os.path.join(data_directory, behs[i]) + '_noisy.txt', 'r') as file:
            for line in file:
                data = list(map(int, line.split()))
                user_id = data[0]
                item_ids = data[1:]
                for item_id in item_ids:
                    # Increment the value for the specific (user_id, item_id)
                    user_item_matrix[user_id, item_id] += beh_rat[i]
    csr = user_item_matrix.tocsr()
    save_npz(f'denoised/{dataset}_inter.npz', csr)
    print("保存成功")
    return user_item_matrix

# 每个行为根据不同的共同交互次数评分
def intra_dataset(behs, data_directory, num_users, num_items, sim_user_mat):
    # Initialize the user-item interaction matrix
    user_item_intra_matrix_list = []

    # Step 2: Populate the matrix
    for i in range(len(behs)):
        user_item_matrix = np.zeros((num_users, num_items), dtype=int)
        with open(os.path.join(data_directory, behs[i]) + '_noisy.txt', 'r') as file:
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
        print("ok")# 记录结束时间
        end_time = time.time()

        # 计算并打印运行时间
        elapsed_time = end_time - start_time
        print(f"运行时间: {elapsed_time:.2f} 秒")
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
    with open(f"{output_pre}{dataset}-{alpha}.pkl", 'wb') as file:
        pickle.dump(user_item_intra_matrix_list, file)        
    return user_item_intra_matrix_list

def normalize(sparse_matrix, beh_rat):
    for i in range(sparse_matrix.shape[0]):
        # 获取第 i 行的非零元素的起始和结束索引
        row_start = sparse_matrix.indptr[i]
        row_end = sparse_matrix.indptr[i+1]
        
        # 如果该行没有非零元素，跳过
        if row_start == row_end:
            continue
        
        # 获取该行的非零元素的值
        row_data = sparse_matrix.data[row_start:row_end]
        
        # 计算最小值和最大值
        row_min = row_data.min()
        row_max = row_data.max()

        # 避免除以0的情况
        if row_max != row_min:
            # 执行归一化到1-6的范围
            sparse_matrix.data[row_start:row_end] = 1 + (row_data - row_min) * (sum(beh_rat)-1 / (row_max - row_min))
        else:
            # 如果最大值等于最小值，归一化结果全为6
            sparse_matrix.data[row_start:row_end] = sum(beh_rat)

    return sparse_matrix


def denoise_low_weight_elements(sparse_matrix, percentage):
    # 获取非零元素的总数
    sparse_matrix_copy = sparse_matrix.copy()
    
    # 获取非零元素的总数
    num_nonzero = sparse_matrix_copy.nnz
    
    # 计算需要去噪的元素数量（20%的非零元素）
    num_denoise = int(num_nonzero * percentage)
    
    # 获取非零元素及其对应的索引
    data = sparse_matrix_copy.data
    
    # 按权重从小到大排序，获取排序后的索引
    sorted_indices = np.argsort(data)
    
    # 选择较小权重的前 num_denoise 个元素的索引
    selected_indices = sorted_indices[:num_denoise]
    
    # 将选中的元素值设为0
    sparse_matrix_copy.data[selected_indices] = 0.0
    
    # 去除值为0的元素并返回稀疏矩阵
    sparse_matrix_copy.eliminate_zeros()

    return sparse_matrix_copy

dataset_list = ["Beibei", "Taobao", "Tmall"]

for dataset in dataset_list:
    
    data_directory = 'dataset/'
    data_directory = data_directory + dataset
    output_pre = 'denoised/'

    if dataset == "Beibei" :
        behs = ['pv', 'cart', 'train']
        beh_rat = [0.42355467272832326, 0.5764453272716766, 1]
        n_user = 21716
        n_item = 7977
        
    elif dataset == "Taobao":
        behs = ['pv', 'cart', 'train']
        beh_rat = [0.4810969603525134, 0.5189030396474865, 1]
        n_user = 48749
        n_item = 39493
        
    elif dataset == "IJCAI":
        beh_rat = [0.5004411263948705, 0.4995588736051295, 0.4995588736051295, 1]
        behs = ['click', 'fav', 'cart', 'train']
        n_user = 17435
        n_item = 35920
        
    elif dataset == "Tmall":
        behs = ['pv', 'fav', 'cart', 'train']
        beh_rat = [0.4800861899155415, 0.5199138100844586, 0.5199138100844586, 1]
        n_user = 31882
        n_item = 31232

    sim_user_mat = sp.load_npz(f"Sim/{dataset}_user_matrix.npz")
    inter_dataset_matrix = inter_dataset(behs, data_directory, n_user, n_item, beh_rat)
    intra_dataset_matrix = intra_dataset(behs, data_directory, n_user, n_item, sim_user_mat)


    # print(f"Saved inter_dataset_matrix inter sparse matrices and intra_dataset_matrix intra sparse matrices.")


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

