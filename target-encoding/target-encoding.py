from collections import defaultdict
def target_encoding(categories: list, targets: list) -> list:
    """
    Returns each category replaced by its mean target.
    """
    hm = defaultdict(list)
    for i,n in enumerate(categories):
        hm[n].append(targets[i])
    res = dict()
    for key in hm:
        res[key] = sum(hm[key]) / len(hm[key])
    for i, cat in enumerate(categories):
        categories[i] = res[cat]
    return categories