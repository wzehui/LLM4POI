import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
import torch.optim as optim
from torch_geometric.nn import ComplEx, DistMult, RotatE, TransE

checkin_data = pd.read_csv('/Users/ze/PycharmProjects/DataCollection/Yelp/checkin.csv')

total_samples = len(checkin_data)
train_size = int(0.7 * total_samples)
validation_size = int(0.1 * total_samples)
test_size = int(0.2 * total_samples)

train_data = checkin_data.iloc[:train_size].copy()
validation_data = checkin_data.iloc[train_size:train_size+validation_size].copy()
test_data = checkin_data.iloc[train_size+validation_size:train_size+validation_size
                                              +test_size].copy()

# Remove cold start user from test data
test_data_cold = test_data[~test_data['user_id'].isin(set(train_data['user_id']))]
test_data = test_data[~test_data['user_id'].isin(set(test_data_cold['user_id']))]

# business = train_data.sort_values(by='business_id')
# business = business.drop_duplicates(subset='business_id', keep='first')
# business = business.reset_index(drop=True)
# business_id_mapping = business['business_id']
# business_map = business_id_mapping.reset_index().set_index('business_id').to_dict()
# train_data['business_id'] = train_data['business_id'].map(business_map['index']).astype(int)
#
# user = train_data.sort_values(by='user_id')
# user = user.drop_duplicates(subset='user_id', keep='first')
# user = user.reset_index(drop=True)
# user_id_mapping = user['user_id']
# user_map = user_id_mapping.reset_index().set_index('user_id').to_dict()
# train_data['user_id'] = train_data['user_id'].map(user_map['index']).astype(int)
#
# mean_business, std_business = 0, 1
# random_business_feature = np.random.normal(mean_business, std_business,
#                                            size=(len(business), 32))
# mean_user, std_user = 0, 1
# random_user_feature = np.random.normal(mean_user, std_user, size=(len(user), 32))
#
# review_data = pd.read_csv('/Users/ze/PycharmProjects/DataCollection/Yelp/review.csv')
# merged_data = pd.merge(train_data, review_data[['review_id', 'stars']],
#                        on='review_id', how='left')
# labels = merged_data['stars']
# y = labels.to_numpy()
#
#
# edge_index = train_data[['user_id', 'business_id']].values.transpose()
# random_edge_feature = np.random.normal(0, 1, size=(edge_index.shape[1], 32))
#
# data = HeteroData()
# data['user'].x = random_user_feature
# data['business'].x = random_business_feature
# data['user', 'review', 'business'].edge_index = edge_index
# data['user', 'review', 'business'].edge_attr = random_edge_feature
# data['user', 'business'].y = y


class GraphProcessor:
    def __init__(self):
        self.review_path = '/Users/ze/PycharmProjects/DataCollection/Yelp/review.csv'

    def process_data(self, input_data):
        # Process business
        business = input_data.sort_values(by='business_id')
        business = business.drop_duplicates(subset='business_id', keep='first')
        business = business.reset_index(drop=True)
        business_id_mapping = business['business_id']
        business_map = business_id_mapping.reset_index().set_index('business_id').to_dict()
        input_data['business_id'] = input_data['business_id'].map(business_map['index']).astype(int)

        # Process user
        user = input_data.sort_values(by='user_id')
        user = user.drop_duplicates(subset='user_id', keep='first')
        user = user.reset_index(drop=True)
        user_id_mapping = user['user_id']
        user_map = user_id_mapping.reset_index().set_index('user_id').to_dict()
        input_data['user_id'] = input_data['user_id'].map(user_map['index']).astype(int)

        # Generate random features
        mean_business, std_business = 0, 1
        random_business_feature = torch.randn(size=(len(business), 32))
        mean_user, std_user = 0, 1
        random_user_feature = torch.randn(size=(len(user), 32))

        # Create edge_index and random_edge_feature
        edge_index = input_data[['user_id', 'business_id']].values.transpose()
        random_edge_feature = torch.randn(size=(edge_index.shape[1], 32))

        # Load review data
        review_data = pd.read_csv(self.review_path)
        merged_data = pd.merge(input_data, review_data[['review_id', 'stars']],
                               on='review_id', how='left')
        labels = merged_data['stars']
        y = torch.tensor(labels.values, dtype=torch.float)

        # Create HeteroData object
        data = HeteroData()
        data['user'].x = random_user_feature
        data['business'].x = random_business_feature
        data['user', 'review', 'business'].edge_index = edge_index
        data['user', 'review', 'business'].edge_attr = random_edge_feature
        data['user', 'business'].y = y

        return data


graph_processor = GraphProcessor()
train_data = graph_processor.process_data(train_data)
val_data = graph_processor.process_data(validation_data)
test_data = graph_processor.process_data(test_data)

model_map = {
    'transe': TransE,
    'complex': ComplEx,
    'distmult': DistMult,
    'rotate': RotatE,
}
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_arg_map = {'rotate': {'margin': 9.0}}
model = model_map['transe'](
    num_nodes=train_data.num_nodes,
    num_relations=len(train_data.edge_types), #train_data.num_edge_types,
    hidden_channels=50,
    **model_arg_map.get('transe', {}),
).to(device)

loader = model.loader(
    head_index=train_data.edge_stores[0].edge_index[0],
    #train_data.edge_index[0],
    rel_type=train_data.edge_types,
    tail_index=train_data.edge_stores[0].edge_index[0],
    #train_data.edge_index[1],
    batch_size=1000,
    shuffle=True,
)

optimizer_map = {
    'transe': optim.Adam(model.parameters(), lr=0.01),
    'complex': optim.Adagrad(model.parameters(), lr=0.001, weight_decay=1e-6),
    'distmult': optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6),
    'rotate': optim.Adam(model.parameters(), lr=1e-3),
}
optimizer = optimizer_map['transe']


def train():
    model.train()
    total_loss = total_examples = 0
    for head_index, rel_type, tail_index in loader:
        optimizer.zero_grad()
        loss = model.loss(head_index, rel_type, tail_index)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * head_index.numel()
        total_examples += head_index.numel()
    return total_loss / total_examples


@torch.no_grad()
def eval(data):
    model.eval()
    return model.test(
        head_index=data.edge_index[0],
        rel_type=data.edge_type,
        tail_index=data.edge_index[1],
        batch_size=20000,
        k=10,
    )


if __name__ == '__main__':
    for epoch in range(1, 501):
        loss = train()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        if epoch % 25 == 0:
            rank, mrr, hits = eval(val_data)
            print(f'Epoch: {epoch:03d}, Val Mean Rank: {rank:.2f}, '
                  f'Val MRR: {mrr:.4f}, Val Hits@10: {hits:.4f}')

    rank, mrr, hits_at_10 = eval(test_data)
    print(f'Test Mean Rank: {rank:.2f}, Test MRR: {mrr:.4f}, '
          f'Test Hits@10: {hits_at_10:.4f}')