import pandas as pd
from torch_geometric.data import HeteroData

anime = pd.read_csv("anime.csv")
rating = pd.read_csv("rating.csv")

# Sort to define the order of nodes
sorted_df = anime.sort_values(by="anime_id").set_index("anime_id")

# Map IDs to start from 0
sorted_df = sorted_df.reset_index(drop=False)
movie_id_mapping = sorted_df["anime_id"]

# Select node features
node_features = sorted_df[["type", "genre", "episodes"]].copy()
# # Convert non-numeric columns
# pd.set_option('mode.chained_assignment', None)

# For simplicity I'll just select the first genre here and ignore the others
genres = node_features["genre"].str.split(",", expand=True)
node_features.loc[:, "main_genre"] = genres[0]

# One-hot encoding
anime_node_features = pd.concat([node_features, pd.get_dummies(node_features["main_genre"])], axis=1, join='inner')
anime_node_features = pd.concat([anime_node_features, pd.get_dummies(anime_node_features["type"])], axis=1, join='inner')
anime_node_features.drop(["genre", "main_genre"], axis=1, inplace=True)

# Convert to numpy
x = anime_node_features.to_numpy()

# Find out mean rating and number of ratings per user
mean_rating = rating.groupby("user_id")["rating"].mean().rename("mean")
num_rating = rating.groupby("user_id")["rating"].count().rename("count")
user_node_features = pd.concat([mean_rating, num_rating], axis=1)

# Remap user ID (to start at 0)
user_node_features = user_node_features.reset_index(drop=False)
user_id_mapping = user_node_features["user_id"]

# Only keep features
user_node_features = user_node_features[["mean", "count"]]

# # Convert to numpy
# x = user_node_features.to_numpy()

rating = rating[~rating["anime_id"].isin([30913, 30924, 20261])].copy()
labels = rating["rating"]
y = labels.to_numpy()

# Map anime IDs
movie_map = movie_id_mapping.reset_index().set_index('anime_id').to_dict()
rating['anime_id'] = rating['anime_id'].map(movie_map['index']).astype(int)
# Map user IDs
user_map = user_id_mapping.reset_index().set_index('user_id').to_dict()
rating['user_id'] = rating['user_id'].map(user_map['index']).astype(int)

edge_index = rating[["user_id", "anime_id"]].values.transpose()

data = HeteroData()
data['user'].x = user_node_features
data['movie'].x = anime_node_features
data['user', 'rating', 'movie'].edge_index = edge_index
data['user', 'movie'].y = y