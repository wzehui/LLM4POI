import os
import os.path as osp
import sys
import re
import json
import pickle
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix


def read_json(file_path, file_name):
    data = []
    with open(os.path.join(file_path, file_name), 'r', encoding='utf-8') as file:
        for line_number, line in tqdm(enumerate(file, start=1),
                                      desc=f"Reading {file_name}"):
            try:
                json_data = json.loads(line)
                data.append(json_data)
            except json.decoder.JSONDecodeError as e:
                print(f"\nError decoding JSON at line {line_number}: {e}")

    return data


def split_json(input_path, file_name, output_path, num_files):
    data = []
    with (open(os.path.join(input_path, file_name), 'r', encoding='utf-8') as file):
        for line_number, line in enumerate(file, start=1):
            if line_number in [1, 2]:
                continue
            try:
                json_data = json.loads(line)
                data.append(json_data)
            except json.decoder.JSONDecodeError as e:
                print(f"Error decoding JSON at line {line_number}: {e}")

    total_records = len(data)
    records_per_file = total_records // num_files

    for i in range(num_files):
        start_index = i * records_per_file
        end_index = (i + 1) * records_per_file if i < num_files - 1 else total_records
        output_name = f"user_{i + 1}.json"

        with open(os.path.join(output_path, output_name), 'w', encoding='utf-8') as output_file:
            json.dump(data[start_index:end_index], output_file, ensure_ascii=False, indent=2)


def reshape_review(data):
    data['reactions'] = data['useful'] + data['funny'] + data['cool']
    checkin_df = data[['review_id', 'user_id', 'business_id', 'stars', 'date']]
    review_df = data[['review_id', 'user_id', 'business_id', 'text', 'stars', 'reactions']]
    return checkin_df, review_df


def merge_with_reviews(data, data_business, data_user):
    data = pd.merge(data, data_business[['business_id', 'name', 'city', 'state',
                                   'postal_code', 'latitude', 'longitude',
                                          'stars', 'review_count',
                                          'attributes', 'categories', 'hours']],
                           on='business_id', how='left')
    data = pd.merge(data, data_user[['user_id', 'name', 'review_count', 'yelping_since', 'reactions',
       'elite', 'friends', 'fans', 'average_stars', 'compliments']],
                    on='user_id', how='left')

    return data


def convert_to_sparse_matrix(data):
    selected_columns = ['user_id', 'review_id', 'business_id', 'date']
    processed_data = data[selected_columns].copy()
    processed_data['review_tuple'] = list(zip(processed_data['review_id'], processed_data['date']))
    user_item_matrix = processed_data.groupby(['user_id', 'business_id'])['review_tuple'].agg(list).unstack()
    user_item_matrix_sparse = csr_matrix(user_item_matrix.notna().astype(int).values)
    sparsity = 1 - user_item_matrix_sparse.nnz / (user_item_matrix_sparse.shape[0] * user_item_matrix_sparse.shape[1])
    return sparsity


def save_csv(data, output_path, file_name):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not file_name.endswith('.csv'):
        file_name += '.csv'
    file_name = re.sub(r'[^\w\s.-_]', '', file_name)
    data.to_csv(os.path.join(output_path, file_name), index=False, encoding='utf-8')


def load_data():
    split_dir = osp.join(osp.dirname(osp.realpath(__file__)),
                         '../data/Yelp_MM/split_data/')
    if not osp.exists(split_dir):
        print('The split files do not exist, please check and put it into: {}'.format(split_dir))
        sys.exit()

    train_data = pd.read_csv(osp.join(split_dir, 'Yelp_MM_train_o.txt'),
                             sep='\t', header=None, names=['user_id',
                                                           'business_id',
                                                           'frequency'])
    test_data = pd.read_csv(osp.join(split_dir, 'Yelp_MM_test_o.txt'),
                            sep='\t', header=None, names=['user_id',
                                                          'business_id',
                                                          'frequency'])
    validation_data = pd.read_csv(osp.join(split_dir, 'Yelp_MM_val_o.txt'),
                                  sep='\t', header=None, names=['user_id',
                                                                'business_id',
                                                                'frequency'])
    return train_data, validation_data, test_data