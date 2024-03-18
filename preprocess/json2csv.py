import os
import sys
import os.path as osp
from utils.proprecess import read_json, convert_to_sparse_matrix, save_csv
from utils.filtering import filter_data
from utils.analyse import visualize_reviews


if __name__ == '__main__':
    json_dir = osp.join(osp.dirname(osp.realpath(__file__)), '../data/Yelp_MM')
    if not osp.exists(json_dir):
        print('The json files does not exist, please download it from '
              'https://www.yelp.com/dataset, and put it into: {}'.format(json_dir))
        sys.exit()

    processed_dir = osp.join(osp.dirname(osp.realpath(__file__)),
                             '../data/Yelp_MM/processed')
    if not osp.exists(processed_dir):
        os.makedirs(processed_dir)
        print('Save Path Created!')

    data_review = read_json(json_dir,
                            'yelp_dataset/yelp_academic_dataset_review.json')
    data_photo = read_json(json_dir, 'yelp_photos/photos.json')
    data_business = read_json(json_dir,
                              'yelp_dataset/yelp_academic_dataset_business.json')
    data_user = read_json(json_dir,
                          'yelp_dataset/yelp_academic_dataset_user.json')

    data_photo, data_business, data_user, data_checkin, data_text = (
        filter_data(data_review, data_photo, data_business, data_user))

    # sparsity = convert_to_sparse_matrix(data_checkin)
    # print("Sparsity matrix:\n", sparsity)
    # visualize_reviews(data_review)

    save_csv(data_text, processed_dir, 'review.csv')
    save_csv(data_checkin, processed_dir, 'checkin.csv')
    save_csv(data_photo, processed_dir, 'photo.csv')
    save_csv(data_business, processed_dir, 'business.csv')
    save_csv(data_user, processed_dir, 'user.csv')
