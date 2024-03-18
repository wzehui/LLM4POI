import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit, ToUndirected
from datetime import datetime
import torch_geometric.transforms as T

checkin_path = '/Users/ze/PycharmProjects/DataCollection/Yelp/checkin.csv'
# checkin_path = '/root/data/Yelp_MM/checkin.csv'



class IdentityEncoder:
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1).to(
            self.dtype)  # torch.from_numpy(df.values).view(-1, 1).to(self.dtype)


def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, **kwargs)
    mapping = {index: i for i, index in enumerate(df.index.unique())}
    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping


def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, **kwargs):
    df = pd.read_csv(path, **kwargs)

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    time = df['date'].apply(
        lambda x: int(datetime.strptime(x, '%Y-%m-%d %H:%M:%S').timestamp()))
    time = torch.tensor([time]).squeeze()
    return edge_index, edge_attr, time


_, user_mapping = load_node_csv(checkin_path, index_col='user_id')
_, business_mapping = load_node_csv(checkin_path, index_col='business_id')

data = HeteroData()
data['user'].num_nodes = len(user_mapping)
data['business'].num_nodes = len(business_mapping)

edge_index, edge_label, time = load_edge_csv(
    checkin_path,
    src_index_col='user_id',
    src_mapping=user_mapping,
    dst_index_col='business_id',
    dst_mapping=business_mapping,
    encoders={'stars': IdentityEncoder(dtype=torch.long)},
)

data['user', 'review', 'business'].edge_index = edge_index
data['user', 'review', 'business'].edge_label = edge_label
data['user', 'review', 'business'].time = time

# # 1. Add a reverse ('movie', 'rev_rates', 'user') relation for message passing.
# data = ToUndirected()(data)
# del data['business', 'rev_review', 'user'].edge_label  # Remove "reverse" label.
#
# # 2. Perform a link-level split into training, validation, and test edges.
# transform = RandomLinkSplit(
#     num_val=0.1,
#     num_test=0.2,
#     neg_sampling_ratio=0.0,
#     edge_types=[('user', 'review', 'business')],
#     rev_edge_types=[('business', 'rev_review', 'user')],
# )
# train_data, val_data, test_data = transform(data)
# print(train_data)
# print(val_data)
# print(test_data)

import argparse
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric import EdgeIndex
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
from torch_geometric.metrics import (
    LinkPredMAP,
    LinkPredPrecision,
    LinkPredRecall,
)
from torch_geometric.nn import MIPSKNNIndex, SAGEConv, to_hetero

parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=20, help='Number of predictions')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Add user node features for message passing:
data['user'].x = torch.eye(data['user'].num_nodes)
del data['user'].num_nodes

# Add business node features for message passing:
data['business'].x = torch.eye(data['business'].num_nodes)
del data['business'].num_nodes

mask = data['user', 'review', 'business'].edge_label >= 4
data['user', 'business'].edge_index = data['user', 'business'].edge_index[:,
                                      mask]
data['user', 'business'].time = data['user', 'business'].time[mask]
del data['user', 'business'].edge_label  # Drop rating information from graph.

# Add a reverse ('business', 'rev_review', 'user') relation for message passing:
data = T.ToUndirected()(data)

# Perform a temporal link-level split into training and test edges:
edge_label_index = data['user', 'business'].edge_index
time = data['user', 'business'].time

perm = time.argsort()
train_index = perm[:int(0.8 * perm.numel())]
test_index = perm[int(0.8 * perm.numel()):]

kwargs = dict(  # Shared data loader arguments:
    data=data,
    num_neighbors=[5, 5, 5],
    batch_size=32,
    time_attr='time',
    num_workers=4,
    persistent_workers=True,
    temporal_strategy='last',
)

train_loader = LinkNeighborLoader(
    edge_label_index=(('user', 'business'), edge_label_index[:, train_index]),
    edge_label_time=time[train_index] - 1,  # No leakage.
    neg_sampling=dict(mode='binary', amount=2),
    shuffle=True,
    **kwargs,
)

# During testing, we sample node-level subgraphs from both endpoints to
# retrieve their embeddings.
# This allows us to do efficient k-NN search on top of embeddings:
src_loader = NeighborLoader(
    input_nodes='user',
    input_time=(time[test_index].min() - 1).repeat(data['user'].num_nodes),
    **kwargs,
)
dst_loader = NeighborLoader(
    input_nodes='business',
    input_time=(time[test_index].min() - 1).repeat(data['business'].num_nodes),
    **kwargs,
)

# Save test edges and the edges we want to exclude when evaluating:
sparse_size = (data['user'].num_nodes, data['business'].num_nodes)
test_edge_label_index = EdgeIndex(
    edge_label_index[:, test_index].to(device),
    sparse_size=sparse_size,
).sort_by('row')[0]
test_exclude_links = EdgeIndex(
    edge_label_index[:, train_index].to(device),
    sparse_size=sparse_size,
).sort_by('row')[0]


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.conv3 = SAGEConv((-1, -1), hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        return x


class InnerProductDecoder(torch.nn.Module):
    def forward(self, x_dict, edge_label_index):
        x_src = x_dict['user'][edge_label_index[0]]
        x_dst = x_dict['business'][edge_label_index[1]]
        return (x_src * x_dst).sum(dim=-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNN(hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = InnerProductDecoder()

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        x_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(x_dict, edge_label_index)


model = Model(hidden_channels=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()

    total_loss = total_examples = 0
    for batch in tqdm(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(
            batch.x_dict,
            batch.edge_index_dict,
            batch['user', 'business'].edge_label_index,
        )
        y = batch['user', 'business'].edge_label

        loss = F.binary_cross_entropy_with_logits(out, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * y.numel()
        total_examples += y.numel()

    return total_loss / total_examples


@torch.no_grad()
def eval(edge_label_index, exclude_links):
    model.eval()

    dst_embs = []
    for batch in dst_loader:  # Collect destination node/movie embeddings:
        batch = batch.to(device)
        emb = model.encoder(batch.x_dict, batch.edge_index_dict)['business']
        emb = emb[:batch['business'].batch_size]
        dst_embs.append(emb)
    dst_emb = torch.cat(dst_embs, dim=0)
    del dst_embs

    # Instantiate k-NN index based on maximum inner product search (MIPS):
    mips = MIPSKNNIndex(dst_emb)

    # Initialize metrics:
    map_metric = LinkPredMAP(k=args.k).to(device)
    precision_metric = LinkPredPrecision(k=args.k).to(device)
    recall_metric = LinkPredRecall(k=args.k).to(device)

    num_processed = 0
    for batch in src_loader:  # Collect source node/user embeddings:
        batch = batch.to(device)

        # Compute user embeddings:
        emb = model.encoder(batch.x_dict, batch.edge_index_dict)['user']
        emb = emb[:batch['user'].batch_size]

        # Filter labels/exclusion by current batch:
        _edge_label_index = edge_label_index.sparse_narrow(
            dim=0,
            start=num_processed,
            length=emb.size(0),
        )
        _exclude_links = exclude_links.sparse_narrow(
            dim=0,
            start=num_processed,
            length=emb.size(0),
        )
        num_processed += emb.size(0)

        # Perform MIPS search:
        _, pred_index_mat = mips.search(emb, args.k, _exclude_links)

        # Update retrieval metrics:
        map_metric.update(pred_index_mat, _edge_label_index)
        precision_metric.update(pred_index_mat, _edge_label_index)
        recall_metric.update(pred_index_mat, _edge_label_index)

    return (
        float(map_metric.compute()),
        float(precision_metric.compute()),
        float(recall_metric.compute()),
    )


for epoch in range(1, 16):
    train_loss = train()
    print(f'Epoch: {epoch:02d}, Loss: {train_loss:.4f}')
    val_map, val_precision, val_recall = eval(
        test_edge_label_index,
        test_exclude_links,
    )
    print(f'Test MAP@{args.k}: {val_map:.4f}, '
          f'Test Precision@{args.k}: {val_precision:.4f}, '
          f'Test Recall@{args.k}: {val_recall:.4f}')