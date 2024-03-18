import pandas as pd
import numpy as np
import torch
from torch_geometric.nn import TransE

if __name__ == '__main__':
    train_data = pd.read_csv('/Users/ze/PycharmProjects/DataCollection/Yelp/Yelp_MM/Yelp_MM_train.txt', sep='\t',
                             header=None,
                             names=['user_id', 'business_id',
                                    'frequency'])
    test_data = pd.read_csv('/Users/ze/PycharmProjects/DataCollection/Yelp/Yelp_MM/Yelp_MM_test.txt', sep='\t', header=None,
                            names=['user_id', 'business_id',
                                   'frequency'])
    validation_data = pd.read_csv('/Users/ze/PycharmProjects/DataCollection/Yelp/Yelp_MM/Yelp_MM_tune.txt', sep='\t',
                                  header=None, names=['user_id',
                                                      'business_id',
                                                      'frequency'])
    combined_train_data = pd.concat([train_data, validation_data],
                                ignore_index=True)
    combined_train_data = combined_train_data.sample(100)

    user_ids = combined_train_data['user_id'].tolist()
    business_ids = combined_train_data['business_id'].tolist()
    frequencies = combined_train_data['frequency'].tolist()

    # 创建图数据
    edge_index = torch.tensor([user_ids, business_ids], dtype=torch.long)
    edge_attr = torch.tensor(frequencies, dtype=torch.float).view(-1, 1)
    # 创建 TransE 模型
    num_nodes = max(max(user_ids), max(business_ids)) + 1
    num_relations = 1  # 在这个例子中，我们只有一种关系
    hidden_channels = 25
    model = TransE(num_nodes=num_nodes, num_relations=num_relations, hidden_channels=hidden_channels)

    # 训练模型
    criterion = torch.nn.MarginRankingLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    batch_size = 32  # 选择适当的批次大小
    num_samples = edge_index.size(1)

    print('finished')

    for epoch in range(100):
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_edge_index = edge_index[:, start:end]
            batch_edge_attr = edge_attr[start:end, :]

            optimizer.zero_grad()

            positive_edge = torch.cat([batch_edge_index.unsqueeze(0), batch_edge_attr.unsqueeze(0)], dim=2)

            # 生成负样本
            negative_edge_attr = -batch_edge_attr
            negative_edge_index = batch_edge_index

            # 随机采样一个负样本
            neg_samples = torch.randint(0, num_nodes, (1, end - start))
            negative_edge_index[1] = neg_samples

            # 转置负样本的边属性
            negative_edge_attr = negative_edge_attr.t()

            # 组合正样本和负样本
            negative_edge = torch.cat([negative_edge_index, negative_edge_attr], dim=1)

            # 计算损失
            loss = model(positive_edge, negative_edge)
            loss.backward()
            optimizer.step()

    print('finished')

    # 获取用户和商家的嵌入向量
    user_embeddings = model.entity_embeddings[:len(user_ids)].detach().numpy()
    business_embeddings = model.entity_embeddings[len(user_ids):].detach().numpy()

    knowledge_graph = {'user_embeddings': dict(zip(user_ids, user_embeddings)),
                       'business_embeddings': dict(zip(business_ids, business_embeddings))}
