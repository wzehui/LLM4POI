import pandas as pd
from tqdm import tqdm
from utils.proprecess import reshape_review
from utils.analyse import analyse_reviews, generate_review_analyse


def filter_data(data_review, data_photo, data_business, data_user):
    review_df = pd.DataFrame(data_review)
    photo_df = pd.DataFrame(data_photo)
    business_df = pd.DataFrame(data_business)
    user_df = pd.DataFrame(data_user)

    review_filtered = review_df[review_df['business_id'].isin(photo_df['business_id'])]
    print('keep business with photos')
    analyse_reviews(review_df, review_filtered)
    review_df = review_filtered

    # Smartphones became popular in 2007
    review_df['date'] = pd.to_datetime(review_df['date'])
    review_filtered = review_df[review_df['date'].dt.year > 2007]
    print('Timespan filtering')
    analyse_reviews(review_df, review_filtered)
    review_df = review_filtered

    oldest_date = review_df['date'].min()
    latest_date = review_df['date'].max()
    print(f"The oldest date is {oldest_date} and the latest date is {latest_date}.\n")

    year_offset = latest_date - pd.DateOffset(years=1)
    recent_business_ids = review_df[review_df['date'] >= year_offset][
        'business_id'].unique()
    review_filtered = review_df[review_df['business_id'].isin(recent_business_ids)]
    print('Active business filtering')
    analyse_reviews(review_df, review_filtered)
    review_df = review_filtered

    business_checkin_counts = review_df.groupby('business_id').size()
    # Delete business with less than xx check-ins
    businesses_to_remove = business_checkin_counts[business_checkin_counts <
                                                   20].index
    review_filtered = review_df[~review_df['business_id'].isin(
        businesses_to_remove)]
    print('Business with min review filtering')
    analyse_reviews(review_df, review_filtered)
    review_df = review_filtered

    checkin_counts = review_df.groupby('user_id').size()
    # Delete users with less than xx check-ins
    users_to_remove = checkin_counts[checkin_counts < 20].index
    review_filtered = review_df[~review_df['user_id'].isin(
        users_to_remove)]
    print('User with min review filtering')
    analyse_reviews(review_df, review_filtered)
    review_df = review_filtered

    checkin_user_max = checkin_counts[checkin_counts > 300].index
    results = []
    for user_id in tqdm(checkin_user_max, desc="Processing Users"):
        above_airplane_speed, above_train_speed = (
            generate_review_analyse(user_id, review_df, business_df))
        results.append(
            {'user_id': user_id, 'above_airplane_speed': above_airplane_speed,
             'above_train_speed': above_train_speed})
    result_df = pd.DataFrame(results)
    filtered_df = result_df[result_df['above_airplane_speed'] > 0]
    review_filtered = review_df[~review_df['user_id'].isin(
        filtered_df['user_id'])]
    print('Remove bot user')
    analyse_reviews(review_df, review_filtered)
    review_df = review_filtered

    # closed_business_id = business_df[business_df['is_open'] == 0]['business_id']
    # review_filtered = review_df[~review_df['business_id'].isin(
    #     closed_business_id)]
    # print('Remove business which is not open')
    # analyse_reviews(review_df, review_filtered)
    # review_df = review_filtered

    user_id_filtered = set(review_df['user_id'].unique()) - set(user_df['user_id'].unique())
    review_filtered = review_df[~review_df['user_id'].isin(user_id_filtered)]
    print('Remove user which not in user.json')
    analyse_reviews(review_df, review_filtered)
    review_df = review_filtered
    checkin_df, text_df = reshape_review(review_df)

    # business filter
    business_df = business_df[business_df['business_id'].isin(review_df['business_id'])]

    photo_ids_grouped = photo_df.groupby('business_id')['photo_id'].agg(
        list).reset_index()

    business_df = pd.merge(business_df, photo_ids_grouped, on='business_id',
                           how='left')

    # photo filter
    photo_df = photo_df[photo_df['business_id'].isin(review_df['business_id'])]
    # file_path = './yelp_dataset/yelp_photos/photos/'
    # destination_dir = './yelp_dataset/yelp_photos/photo_MM/'
    # if not os.path.exists(destination_dir):
    #     os.makedirs(destination_dir)
    #
    # for index, row in data_photo.iterrows():
    #     photo_id = row['photo_id']
    #     destination_path = os.path.join(destination_dir, f'{photo_id}.jpg')
    #
    #     shutil.copy2(file_path, destination_path)
    #
    # print(f"photos are saved under {destination_dir}")

    # user filter
    user_df = user_df[user_df['user_id'].isin(review_df['user_id'])]
    existing_user_id = set(user_df['user_id'])
    user_df['original_friends_count'] = user_df['friends'].apply(
        lambda x: len(x.split(', ')))
    user_df['friends'] = user_df['friends'].apply(lambda x: ', '.join(
        user for user in x.split(', ') if user in existing_user_id))
    user_df['filtered_friends_count'] = user_df['friends'].apply(
        lambda x: len(x.split(', ')))
    print(f"friend number from {user_df['original_friends_count'].mean()} to "
          f"{user_df['filtered_friends_count'].mean()}")

    user_df['reactions'] = (user_df['useful'] + user_df['funny'] +
                            user_df['cool'])
    user_df['compliments'] = (user_df['compliment_hot'] + user_df['compliment_more'] +
                              user_df['compliment_profile'] + user_df[
                                  'compliment_cute'] + user_df['compliment_list'] +
                              user_df['compliment_note'] + user_df['compliment_plain']
                              + user_df['compliment_cool'] + user_df[
                                  'compliment_funny'] + user_df['compliment_writer'] +
                              user_df['compliment_photos'])
    user_df = user_df[['user_id', 'name', 'review_count', 'yelping_since',
                       'reactions', 'elite', 'friends', 'fans', 'average_stars',
                       'compliments']]

    return photo_df, business_df, user_df, checkin_df, text_df

