import optuna
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

import sys
sys.path.insert(0, '/workspace/LLM4POI/')
from utils.metric import precision_recall_ndcg_at_k
from utils.proprecess import load_data

if __name__ == '__main__':
    train_data, val_data, test_data = load_data()
    combined_train_data = pd.concat([train_data, val_data],
                                    ignore_index=True)

    train_sparse_matrix = csr_matrix((combined_train_data['frequency'],
                                      (combined_train_data['user_id'],
                                       combined_train_data['business_id'])))


    def objective(trial):
        factors = trial.suggest_int('factors', 50, 200)
        regularization = trial.suggest_float('regularization', 0.01, 0.1)
        alpha = trial.suggest_float('alpha', 1, 20)

        model = AlternatingLeastSquares(factors=factors,
                                        regularization=regularization,
                                        alpha=alpha, iterations=50,
                                        random_state=2024)

        model.fit(train_sparse_matrix)

        k = 20
        [pre, rec, map_, ndcg] = [[] for j in range(4)]

        for user_id in tqdm(set(test_data['user_id'])):
            ids, scores = model.recommend(user_id, train_sparse_matrix[user_id],
                                          N=k, filter_already_liked_items=False)

            label = test_data[test_data['user_id'] == user_id]['business_id']
            precision, recall, map__, ndcg_ = (precision_recall_ndcg_at_k(k=k, rankedlist=ids,
                                                  test_matrix=label))

            pre.append(precision),
            rec.append(recall),
            map_.append(map__),
            ndcg.append(ndcg_)

        return sum(pre) / len(pre)


    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    print(f"Best Parameters: {best_params}")

    best_precision = study.best_value
    print(f"Best Precision at K=20: {best_precision}")
