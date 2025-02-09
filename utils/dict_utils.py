def get_first_k(d, k=5):
    keylist = list(d.keys())[:k]
    return {k:d[k] for k in keylist}

def dict_to_string(d, k=5):
    d = get_first_k(d, k)
    return '|'.join(f"{k}:{v:.2f}" for k,v in d.items())