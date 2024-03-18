import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path as osp

visualization_dir = processed_dir = osp.join(osp.dirname(osp.realpath(__file__)),
                              '../results/visualization/')


def analyse_reviews(reviews_df, reviews_filtered):
    reviews_count_filtered = len(reviews_filtered)
    users_count_filtered = reviews_filtered['user_id'].nunique()
    business_count_filtered = reviews_filtered['business_id'].nunique()

    total_reviews_count = len(reviews_df)
    total_users_count = reviews_df['user_id'].nunique()
    total_business_count = reviews_df['business_id'].nunique()

    percentage_reviews_filtered = (reviews_count_filtered / total_reviews_count) * 100
    percentage_users_filtered = (users_count_filtered / total_users_count) * 100
    percentage_business_filtered = (business_count_filtered / total_business_count) * 100

    checkin_counts = reviews_df.groupby('user_id').size()
    checkin_counts_filtered = reviews_filtered.groupby('user_id').size()

    print(f"Review counts: {total_reviews_count:.0f} -->"
          f" {reviews_count_filtered:.0f}")
    print(f"User counts: {total_users_count:.0f} -->"
          f" {users_count_filtered:.0f}")
    print(f"Business counts: {total_business_count:.0f} -->"
          f" {business_count_filtered:.0f}")
    print(f"Percentage of reviews retained: {percentage_reviews_filtered:.2f}%")
    print(f"Percentage of users retained: {percentage_users_filtered:.2f}%")
    print(f"Percentage of businesses retained: "
          f"{percentage_business_filtered:.2f}%")
    print(f"Average sequential length: "
          f"{checkin_counts.mean():.1f} --> "
          f"{checkin_counts_filtered.mean():.1f}")
    print(f"max/min sequential length: {checkin_counts_filtered.max():.0f} / "
          f"{checkin_counts_filtered.min():.0f} \n")


def visualize_reviews(reviews_df):
    checkin_counts = reviews_df.groupby('user_id').size()
    step = 1
    bins = range(min(checkin_counts), 300, step)
    plt.figure()
    plt.hist(checkin_counts, bins=bins)
    plt.xlabel('Check-in Count')
    plt.ylabel('Frequency')
    plt.title('Distribution of Count')
    plt.savefig(visualization_dir, bbox_inches='tight', dpi=300)

    monthly_distribution = reviews_df.resample('M', on='date').size()
    plt.figure()
    plt.bar(monthly_distribution.index, monthly_distribution.values, width=40)
    plt.title('Review Date Distribution')
    plt.xlabel('Date')
    plt.ylabel('Number of Reviews')
    plt.savefig(visualization_dir, bbox_inches='tight', dpi=300)


def visualize_speed(data, fastest_airplane_speed, fastest_train_speed):
    plt.plot(data['date'], data['speed'], linestyle='-',
             color='skyblue')
    plt.title('Checkin Speed Over Time')
    plt.xlabel('Check-in Time')
    plt.ylabel('Speed (km/h)')

    plt.axhline(y=fastest_airplane_speed, color='red', linestyle='--',
                label='Fastest Airplane')
    plt.axhline(y=fastest_train_speed, color='green', linestyle='--',
                label='Fastest Train')

    plt.legend()
    plt.savefig(visualization_dir, dpi=300, bbox_inches='tight')
    # plt.show()


def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(
            np.radians(lat2)) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = R * c

        return distance


def generate_review_analyse(user_id, reviews_df, business_df):
    fastest_airplane_speed = 900
    fastest_train_speed = 400
    data = reviews_df[reviews_df['user_id'] == user_id]
    data = data.sort_values(by=['date'])

    data = pd.merge(data, business_df[['business_id', 'name', 'city', 'state',
                                   'postal_code', 'latitude', 'longitude',
                                          'stars', 'review_count',
                                          'attributes', 'categories', 'hours']],
                           on='business_id', how='left')
    data['distance'] = haversine(data['latitude'].shift(),
                                 data['longitude'].shift(), data['latitude'],
                                 data['longitude'])
    data['time_diff'] = (data['date'] - data['date'].shift()).dt.total_seconds() / 3600
    data['speed'] = data['distance'] / data['time_diff']
    above_airplane_speed = ((data['speed'] >
                                     fastest_airplane_speed).sum() / len(data['speed']))
    above_train_speed = ((data['speed'] >
                             fastest_train_speed).sum() / len(data['speed']))
    # visualize_speed(data, fastest_airplane_speed, fastest_train_speed)

    return above_airplane_speed, above_train_speed

