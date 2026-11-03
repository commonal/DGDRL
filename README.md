# DGDRL — Selected Core Implementation

Partial code release for **Dual-Guided Debiased Representation Learning Framework for Recommender Systems**.

**The complete code will be released upon acceptance of the paper.** This initial release provides selected grouping, alignment, and contrastive-learning routines for inspection. It is not an end-to-end reproduction package.

## Included routines

| File | Content |
| --- | --- |
| `dgdrl/grouping.py` | Training-neighborhood sorting, equal-size neighbor pairing on both sides, and pair-index caching |
| `dgdrl/batch_grouping.py` | Batch occurrence sorting and splitting using training interaction counts |
| `dgdrl/alignment.py` | Pairwise alignment of normalized embeddings |
| `dgdrl/contrastive.py` | Within-group and weighted cross-group contrastive similarities |
| `example.py` | Small CPU example with synthetic interactions, cache reuse, and backward propagation |

The full encoder, preference-weighted ranking integration, training entry points, dataset preparation, and evaluation pipeline are reserved for the complete release.

## Quick start

```bash
pip install -r requirements.txt
python example.py
```

This example checks the released routines on synthetic inputs; it does not reproduce the paper's experiment tables.

## Inputs and use

The grouping routines accept an object with `train_U2I`, `train_I2U`, `pop_train_count`, `active_train_count`, and `dataset_name`. Neighborhood IDs index the corresponding interaction-count arrays. `get_groups` returns item-pair arrays `(G1, G2)` followed by user-pair arrays `(G3, G4)`. Each pair combines corresponding positions from the lower and upper sorted halves. For odd neighborhood sizes, one neighbor is randomly excluded from pair construction. Cache the resulting indices and reuse them during training.

`split_bacth_items` retains its historical function name. It splits sorted batch occurrences before the training caller deduplicates each half. `InfoNCE_i` uses two matching augmented views of one group and the second augmented view of the other group. Its `gama` argument weights cross-group similarities. The exported loss expression is preserved as implemented, including its additional positive-score denominator term. `alignment_user` is applied separately to the item pairs and user pairs, and the two losses are added. Inputs to this example have nonempty groups and alignment pairs; no new empty-group fallback is introduced by this release.

## Source and acknowledgments

These routines were extracted from the frozen DGDRL server source used in the revision work. Function bodies are retained; imports, formatting, and explanatory docstrings have been organized for this partial release.

The implementation builds on [PAAC: Popularity-Aware Alignment and Contrast for Mitigating Popularity Bias](https://github.com/miaomiao-cai2/KDD2024-PAAC), with bilateral grouping/alignment and user-side contrastive integration in DGDRL. We acknowledge the original authors and retain the applicable upstream rights. This repository does not relicense third-party contributions.
