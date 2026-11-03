"""Synthetic CPU demonstration of the selected routines."""
import random
import tempfile
from types import SimpleNamespace
import numpy as np
import torch
from dgdrl.grouping import get_groups
from dgdrl.batch_grouping import split_bacth_items
from dgdrl.alignment import alignment_user
from dgdrl.contrastive import InfoNCE_i

random.seed(12345)
np.random.seed(12345)
torch.manual_seed(12345)
u2i = {0: [0, 1, 2], 1: [0, 1], 2: [0, 2, 3], 3: [1, 2, 3]}
i2u = {i: [u for u, items in u2i.items() if i in items] for i in range(4)}
data = SimpleNamespace(train_U2I=u2i, train_I2U=i2u,
    pop_train_count=np.array([len(i2u[i]) for i in range(4)]),
    active_train_count=np.array([len(u2i[u]) for u in range(4)]),
    dataset_name="synthetic")
with tempfile.TemporaryDirectory() as cache:
    groups = get_groups(data, cache_dir=cache)
    cached = get_groups(data, cache_dir=cache)
    assert all(np.array_equal(a, b) for a, b in zip(groups, cached))
g1, g2, g3, g4 = [torch.as_tensor(g, dtype=torch.long) for g in groups]
assert len(g1) == len(g2) == 4 and len(g3) == len(g4) == 4
users = torch.randn(4, 8, requires_grad=True)
items = torch.randn(4, 8, requires_grad=True)
alignment = alignment_user(items[g1], items[g2]) + alignment_user(users[g3], users[g4])
indices = torch.arange(4)
lo, hi = split_bacth_items(indices, data.pop_train_count)
lo, hi = torch.as_tensor(np.unique(lo)), torch.as_tensor(np.unique(hi))
view1 = items + 0.01 * torch.rand_like(items)
view2 = items + 0.01 * torch.rand_like(items)
contrastive = 0.5 * InfoNCE_i(view1[lo], view2[lo], view2[hi], 0.2, 0.6)
contrastive += 0.5 * InfoNCE_i(view1[hi], view2[hi], view2[lo], 0.2, 0.6)
loss = alignment + contrastive
assert torch.isfinite(loss)
loss.backward()
assert torch.isfinite(users.grad).all() and torch.isfinite(items.grad).all()
print(f"Core example passed: alignment={alignment.item():.6f}, contrastive={contrastive.item():.6f}")
