def get_first_k(d, k=5):
    keylist = list(d.keys())[:k]
    return {k:d[k] for k in keylist}