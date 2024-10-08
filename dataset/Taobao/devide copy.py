import os
import random
from collections import defaultdict

def read_data(file_path):
    data = defaultdict(list)
    with open(file_path, 'r') as file:
        for line in file:
            items = line.strip().split()
            user_id = items[0]
            item_ids = items[1:]
            data[user_id].extend(item_ids)
    return data

def split_data(data, train_ratio=0.7, valid_ratio=0.3):
    train_data = defaultdict(list)
    valid_data = defaultdict(list)
    
    for user_id, item_ids in data.items():
        random.shuffle(item_ids)
        train_end = int(len(item_ids) * train_ratio)
        train_data[user_id] = item_ids[:train_end]
        valid_data[user_id] = item_ids[train_end:]
    
    return train_data, valid_data

def write_data(data, file_path, fill_zero=False):
    with open(file_path, 'w') as file:
        for user_id, item_ids in data.items():
            if fill_zero:
                item_ids = ['0'] * len(item_ids)
            file.write(f"{user_id} {' '.join(item_ids)}\n")

def main():
    current_folder = os.path.dirname(os.path.abspath(__file__))
    
    for interaction_type in ['pv', 'cart', 'train']:
        input_file = os.path.join(current_folder, f"{interaction_type}.txt")
        data = read_data(input_file)
        
        if interaction_type == 'train':
            train_data, valid_data = split_data(data)
            output_type_folder = os.path.join(current_folder, interaction_type)
            os.makedirs(output_type_folder, exist_ok=True)
            write_data(train_data, os.path.join(output_type_folder, 'train.txt'))
            write_data(valid_data, os.path.join(output_type_folder, 'valid.txt'))
        else:
            output_type_folder = os.path.join(current_folder, interaction_type)
            os.makedirs(output_type_folder, exist_ok=True)
            write_data(data, os.path.join(output_type_folder, 'test.txt'), fill_zero=True)
            write_data(data, os.path.join(output_type_folder, 'valid.txt'), fill_zero=True)

if __name__ == "__main__":
    main()
