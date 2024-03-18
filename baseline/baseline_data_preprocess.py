import os
import sys
import pickle
import pandas as pd
import os.path as osp

if __name__ == '__main__':
    def mapping_id(result):
        # Group by user_id, business_id, and date, then count the occurrences
        result = result.groupby(
            ['user_id', 'business_id', 'date']).size().reset_index(
            name='frequency')
        result = result.sort_values(by='date')
        result['user_id_numeric'], user_id_labels = pd.factorize(
            result['user_id'])
        result['business_id_numeric'], business_id_labels = pd.factorize(
            result['business_id'])
        result = result.drop(['user_id', 'business_id'], axis=1)

        return result

    split_dir = osp.join(osp.dirname(osp.realpath(__file__)),
                         '../data/Yelp_MM/split')
    if not osp.exists(split_dir):
        print('The processed csv files do not exist, please check and put it '
              'into: {}'.format(split_dir))
        sys.exit()

    baseline_data_dir = osp.join(osp.dirname(osp.realpath(__file__)),
                                 '../data/Yelp_MM/baseline_data/')
    if not osp.exists(baseline_data_dir):
        os.makedirs(split_dir)
        print('Save Path Created!')

    train_data = pd.read_csv(osp.join(split_dir, 'Yelp_MM_train.csv'))

    train_size = int(7/8 * len(train_data))
    validation_size = int(1/8 * len(train_data))
    validation_data = train_data.iloc[train_size:train_size + validation_size]
    train_data = train_data.iloc[:train_size]
    test_data = pd.read_csv(osp.join(split_dir, 'Yelp_MM_test.csv'))

    train_data = mapping_id(train_data)
    validation_data = mapping_id(validation_data)
    test_data = mapping_id(test_data)

    # Save the splits to separate files or further process as needed
    train_data.to_csv(osp.join(baseline_data_dir, 'Yelp_MM_train.txt'), sep='\t',
                      index=False,
                      header=False, columns=['user_id_numeric',
                                             'business_id_numeric',
                                             'frequency'])
    validation_data.to_csv(osp.join(baseline_data_dir, 'Yelp_MM_val.txt'), sep='\t',
                           index=False,
                           header=False, columns=['user_id_numeric',
                                                  'business_id_numeric',
                                                  'frequency'])
    test_data.to_csv(osp.join(baseline_data_dir, 'Yelp_MM_test.txt'), sep='\t',
                     index=False,
                     header=False, columns=['user_id_numeric',
                                            'business_id_numeric',
                                            'frequency'])