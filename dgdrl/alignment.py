# Selected DGDRL implementation; PAAC-derived routines are acknowledged in README.md.
import torch.nn.functional as F

def alignment_user(x, y):
    """Mean squared Euclidean distance between corresponding L2-normalized vectors; call separately on item pairs and user pairs."""
    x, y = (F.normalize(x, dim=-1), F.normalize(y, dim=-1))
    return (x - y).norm(p=2, dim=1).pow(2).mean()
