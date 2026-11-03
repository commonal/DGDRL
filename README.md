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


