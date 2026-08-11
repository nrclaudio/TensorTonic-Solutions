def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    hits = [i for i in recommended[:k] if i in relevant]
    precision_k = len(hits) / k
    recall_k = len(hits) / len(relevant)
    return [precision_k, recall_k]