# Selected DGDRL implementation; PAAC-derived routines are acknowledged in README.md.
import os
import random
import numpy as np

def user_items_2_group_pop(data):
    """Return flattened low/high-popularity item pairs within user neighborhoods. Odd neighborhoods randomly discard one neighbor for pairing only."""
    G1, G2 = ([], [])
    for u in data.train_U2I.keys():
        items = data.train_U2I[u]
        items_sorted = list(np.array(items)[np.argsort(np.array(data.pop_train_count)[items])])
        if len(items) % 2 != 0:
            items_sorted = np.delete(items_sorted, random.sample(range(len(items_sorted)), 1)[0])
        num = int(len(items_sorted) / 2)
        G1.extend(items_sorted[0:num])
        G2.extend(items_sorted[num:])
    return (np.array(G1), np.array(G2))

def item_users_2_group_act(data, min_users=1):
    """Return flattened low/high-activity user pairs within item neighborhoods. The first array contains lower-activity users."""
    G1, G2 = ([], [])
    for i in data.train_I2U.keys():
        users = data.train_I2U[i]
        if len(users) < min_users:
            continue
        users_sorted = list(np.array(users)[np.argsort(np.array(data.active_train_count)[users])])
        if len(users_sorted) % 2 != 0:
            users_sorted.pop(random.choice(range(len(users_sorted))))
        num = len(users_sorted) // 2
        G1.extend(users_sorted[:num])
        G2.extend(users_sorted[num:])
    return (np.array(G1), np.array(G2))

def save_groups(G1, G2, G3, G4, dataset_name='default', cache_dir='cache'):
    """Save item-pair and user-pair indices to a dataset-specific cache."""
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.join(cache_dir, f'{dataset_name}_saved_groups.npz')
    np.savez(filename, G1=G1, G2=G2, G3=G3, G4=G4)

def load_groups(dataset_name='default', cache_dir='cache'):
    """Load the four cached alignment-index arrays."""
    filename = os.path.join(cache_dir, f'{dataset_name}_saved_groups.npz')
    if not os.path.exists(filename):
        raise FileNotFoundError(f'No cached file found for dataset: {dataset_name}')
    loaded = np.load(filename)
    return (loaded['G1'], loaded['G2'], loaded['G3'], loaded['G4'])

def get_groups(data, force_reload=False, cache_dir='cache'):
    """Load cached pair indices, or construct and cache them from training neighborhoods."""
    dataset_name = data.dataset_name
    filename = os.path.join(cache_dir, f'{dataset_name}_saved_groups.npz')
    if not force_reload and os.path.exists(filename):
        print(f'Loading cached group data for dataset: {dataset_name}...')
        return load_groups(dataset_name, cache_dir)
    print(f'Generating new group data for dataset: {dataset_name}...')
    G1, G2 = user_items_2_group_pop(data)
    G3, G4 = item_users_2_group_act(data)
    save_groups(G1, G2, G3, G4, dataset_name, cache_dir)
    return (G1, G2, G3, G4)
