import math

def precision_recall_ndcg_at_k(k, rankedlist, test_matrix):
    idcg_k = 0
    dcg_k = 0
    map = 0
    ap = 0

    b1 = rankedlist
    b2 = test_matrix
    s2 = set(b2)
    hits = [(idx, val) for idx, val in enumerate(b1) if val in s2]
    count = len(hits)

    for c in range(count):
        ap += (c + 1) / (hits[c][0] + 1)
        dcg_k += 1 / math.log(hits[c][0] + 2, 2)

    for i in range(count):
        idcg_k += 1 / math.log(i + 2, 2)

    ndcg_k = dcg_k / idcg_k if idcg_k != 0 else 0

    return (float(count / k), float(count / len(test_matrix)), float(ap / k),
            float(ndcg_k))

