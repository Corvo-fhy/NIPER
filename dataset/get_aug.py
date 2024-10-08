import random


def get_random_subset_by_percentage(original_set, percentage):
    # 计算要选取的元素数量
    sample_size = max(1, int(len(original_set) * percentage))
    
    # 从原集合中随机选取指定数量的元素，并将结果转换为集合返回
    return set(random.sample(original_set, sample_size))


def get_aug(dataset):
    user_interactions = {i: {} for i in range(len(behs))}
    for i in range(len(behs)):
        with open(f"{dataset}/{behs[i]}.txt", 'r') as f:
            for line in f:
                parts = line.strip().split()
                user_id = parts[0]
                item_ids = set(parts[1:])
                if user_id in user_interactions[i]:
                    user_interactions[i][user_id].update(item_ids)
                else:
                    user_interactions[i][user_id] = item_ids
    augmented_train = {}

    for user_id in user_interactions[-1].keys():
        aux_items = set()
        for i in range(len(behs) - 1):
            items = user_interactions[i].get(user_id, set())
            aux_items.update(items)
        train_items = user_interactions[-1].get(user_id)
        
        # 更新 train 数据
        random_subset = get_random_subset_by_percentage(aux_items, 0.5)
        augmented_train[user_id] = train_items | aux_items

    # 保存增强后的数据到 aug.txt
    with open('aug.txt', 'w') as f:
        for user_id, item_ids in augmented_train.items():
            f.write(f"{user_id} {' '.join(item_ids)}\n")

    print("增强后的文件已保存为 aug.txt")


dataset = "IJCAI"
if dataset == "Beibei" :
    behs = ['pv', 'cart', 'train']
    beh_rat = [1/6, 1/3, 1/2]
    n_user = 21716
    n_item = 7977
    
elif dataset == "Taobao":
    behs = ['pv', 'cart', 'train']
    beh_rat = [1/6, 1/3, 1/2]
    n_user = 48749
    n_item = 39493
    
elif dataset == "IJCAI":
    beh_rat = [1/8, 3/16, 3/16, 1/2]
    behs = ['click', 'fav', 'cart', 'train']
    n_user = 17435
    n_item = 35920
    
elif dataset == "Tmall":
    behs = ['pv', 'fav', 'cart', 'train']
    beh_rat = [1/8, 3/16, 3/16, 1/2]
    n_user = 31882
    n_item = 31232
    
    
get_aug(dataset)