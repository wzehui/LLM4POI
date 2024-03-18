import pandas as pd


checkin_data = pd.read_csv('/Users/ze/PycharmProjects/DataCollection/Yelp/checkin.csv')
# Group by user_id, business_id, and date, then count the occurrences
checkin_data = checkin_data.groupby(['user_id', 'business_id', 'date']).size().reset_index(name='frequency')
checkin_data = checkin_data.sort_values(by='date')

total_samples = len(checkin_data)
train_size = int(0.7 * total_samples)
validation_size = int(0.1 * total_samples)
test_size = int(0.2 * total_samples)

train_data = checkin_data.iloc[:train_size]
validation_data = checkin_data.iloc[train_size:train_size+validation_size]
test_data = checkin_data.iloc[train_size+validation_size:train_size+validation_size
                                              +test_size]
# Remove cold start user from test data
test_data_cold = test_data[~test_data['user_id'].isin(set(train_data['user_id']) | set(validation_data['user_id']))]
test_data = test_data[~test_data['user_id'].isin(set(test_data_cold['user_id']))]

# Save the splits to separate files or further process as needed
train_data.to_csv('./Yelp_MM_train.txt', sep='\t', index=False,
                  header=False, columns=['user_id',
                                         'business_id',
                                         'frequency'])
test_data.to_csv('./Yelp_MM_test.txt', sep='\t', index=False,
                 header=False, columns=['user_id',
                                        'business_id', 'frequency'])
validation_data.to_csv('./Yelp_MM_tune.txt', sep='\t', index=False,
                       header=False, columns=['user_id',
                                              'business_id',
                                              'frequency'])

checkin_data = pd.concat([train_data, test_data, validation_data], ignore_index=True)
user_entities = set(checkin_data['user_id'].unique())
business_entities = set(checkin_data['business_id'].unique())
# Find common entities between 'user_id' and 'business_id'
common_entities = user_entities.intersection(business_entities)
# Add "_poi" suffix to 'business_id' values that are common with 'user_id'
checkin_data.loc[checkin_data['business_id'].isin(common_entities), 'business_id'] += '_poi'

# Extract unique entities (user_id and business_id)
entities = pd.concat([checkin_data['user_id'], checkin_data['business_id']]).unique()
entity_labels, entity_letter = pd.factorize(entities)
# Generate entity.dict
entity_dict = pd.DataFrame({'entity_id': entity_labels, 'entity': entity_letter})
entity_dict.to_csv('./entity.dict', sep='\t', index=False, header=None)

# Extract unique relations
relations = ['review']
# Generate relation.dict
relation_dict = pd.DataFrame({'relation_id': range(len(relations)), 'relation': relations})
relation_dict.to_csv('./relation.dict', sep='\t', index=False, header=None)
