import os
import sys
import pandas as pd
import os.path as osp
from utils.proprecess import save_csv

if __name__ == '__main__':
    csv_dir = osp.join(osp.dirname(osp.realpath(__file__)),
                        '../data/Yelp_MM/processed')
    if not osp.exists(csv_dir):
        print('The processed csv files do not exist, please check and put it '
              'into: {}'.format(csv_dir))
        sys.exit()

    split_dir = osp.join(osp.dirname(osp.realpath(__file__)),
                            '../data/Yelp_MM/split/')
    if not osp.exists(split_dir):
        os.makedirs(split_dir)
        print('Save Path Created!')

    checkin_data = pd.read_csv(osp.join(csv_dir, 'checkin.csv'))
    checkin_data = checkin_data.sort_values(by='date', ascending=True)

    total_samples = len(checkin_data)
    train_size = int(0.8 * total_samples)
    test_size = int(0.2 * total_samples)

    train_data = checkin_data.iloc[:train_size]
    test_data = checkin_data.iloc[train_size:train_size+test_size]

    # Keep four or more stars for training data
    train_data= train_data[train_data['stars'] >= 4]

    # Remove cold start user from test data
    test_data_cold = test_data[~test_data['user_id'].isin(set(train_data['user_id']))]
    test_data = test_data[~test_data['user_id'].isin(set(test_data_cold['user_id']))]

    # Save the splits to separate files or further process as needed
    save_csv(train_data, split_dir, 'Yelp_MM_train.csv')
    save_csv(test_data, split_dir, 'Yelp_MM_test.csv')
