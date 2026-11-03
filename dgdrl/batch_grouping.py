# Selected DGDRL implementation; PAAC-derived routines are acknowledged in README.md.
import numpy as np

def split_bacth_items(items, popular):
    """Sort batch occurrences by training interaction count and split at floor(n/2). Historical spelling is retained. Caller deduplicates each half separately."""
    G1, G2 = ([], [])
    items_sorted = list(np.array(items)[np.argsort(np.array(popular)[items])])
    num = int(len(items_sorted) / 2)
    G1.extend(items_sorted[0:num])
    G2.extend(items_sorted[num:])
    return (np.array(G1), np.array(G2))
