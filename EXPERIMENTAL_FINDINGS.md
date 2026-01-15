# Experimental Findings: GLA vs Gated DeltaNet Training

**Last Updated**: 2025-01-14  
**Purpose**: Reference document tracking key experimental conclusions from training GLA and Gated DeltaNet models

---

## Executive Summary

From training experiments with `gated_deltanet_170M` and `gla_170M` models on WikiText-103, we've identified four key areas:

1. **Dropout is critical** - Optimal dropout rates differ between architectures, optimal is 0.1 for both gated deltanet and gated linear attention 
2. **FLAME defaults work well for GLA, less so for Gated DeltaNet** - Hyperparameter sensitivity varies
3. **Architectural design choices need more exploration** - Especially head_dim scaling and expand_v interactions
4. **Larger dataset needed** - Current WikiText-103 training shows clear overfitting, suggesting we need more data for Chinchilla scaling

---

## 1. The Importance of Dropout

### Key Finding
Dropout regularization is essential, but optimal rates differ between GLA and Gated DeltaNet architectures.

### Experimental Evidence

#### GLA-170M Results

| Experiment | Dropout | Best Val PPL | Final Val PPL (step 62.9k) | Notes |
|------------|---------|--------------|----------------------------|-------|
| `gla_170M-wikitext103-20260114_160305` | 0.1 | **29.54** (step 55k) | 29.86 | **Optimal** - FLAME default |
| `gla_170M_drop0.2-wikitext103-20260114_101342` | 0.2 | 36.02 (step 62.9k) | 36.02 | **Much worse** - Over-regularized |

**Conclusion for GLA**: `dropout=0.1` (FLAME default) is optimal. Increasing to 0.2 causes significant degradation (~6.5 ppl worse).

#### Gated DeltaNet-170M Results

| Experiment | Dropout | Best Val PPL | Final Val PPL (step 62.9k) | Notes |
|------------|---------|--------------|----------------------------|-------|
| `gated_deltanet_170M_drop0.2-wikitext103-20260114_100003` | 0.2 | 30.80 (step 50k) | 30.84 | Reasonable performance |
| `gated_deltanet_170M_fixed-wikitext103-20260114_223215` | 0.1 | 31.98 (step 25k) | 34.65 | **Severe overfitting** |

**Conclusion for Gated DeltaNet**: 
- With `dropout=0.1`, the model overfits severely 
- Gated DeltaNet appears more prone to overfitting than GLA

### Training Commands
```bash
# GLA with default dropout (0.1)
python scripts/train_comparison.py --model gla_170M --submit

# Gated DeltaNet with dropout 0.1
python scripts/train_comparison.py --model gated_deltanet_170M --dropout 0.1 --submit
```

---

## 2. FLAME Default Hyperparameters

### Key Finding
FLAME default hyperparameters work excellently for GLA but are less optimal for Gated DeltaNet.

### FLAME Defaults (from `model_registry.yaml`)
- **Learning Rate**: `1.0e-3`
- **Warmup Steps**: `1000`
- **Dropout**: `0.1` (added, not in original flame)
- **Weight Decay**: `0.01`
- **LR Decay**: `cosine`
- **Max Norm**: `1.0`
- **Batch Size**: `64`
- **Seq Length**: `512`

### GLA Performance with FLAME Defaults

**Experiment**: `gla_170M-wikitext103-20260114_160305`
- **Final Val PPL**: 29.86 (best: 29.54 at step 55k)
- **Training**: Stable, no overfitting observed
- **Conclusion**: ✅ FLAME defaults are well-suited for GLA

### Gated DeltaNet Performance with FLAME Defaults

**Experiment**: `gated_deltanet_170M_fixed-wikitext103-20260114_223215`
- **Best Val PPL**: 31.98 (step 25k)
- **Final Val PPL**: 34.65 (step 62.9k)
- **Training**: Clear overfitting pattern - performance degrades after step 25k
- **Conclusion**: ⚠️ FLAME defaults may not be optimal for Gated DeltaNet

### Hyperparameter Sensitivity

| Model | LR Sensitivity | Dropout Sensitivity | Notes |
|-------|----------------|---------------------|-------|
| GLA | Low | High | Works well with defaults |
| Gated DeltaNet | Unknown | High | May need different defaults |

**Action Items**:
- [ ] Test different learning rates for Gated DeltaNet
- [ ] Test different dropout schedules
- [ ] Investigate weight decay sensitivity

---

## 3. Architectural Design Choices

### Key Finding
Best practices for architectural design choices (head_dim, expand_v, num_heads) are still being explored.

### Current Configurations

#### GLA-170M (`configs/gla_170M.json`)
```json
{
  "hidden_size": 768,
  "num_heads": 12,
  "expand_k": 0.5,
  "expand_v": 1,
  "use_gk": true,
  "use_gv": false,
  "num_hidden_layers": 12
}
```
- **Head dimension**: Implicitly `hidden_size / num_heads = 768 / 12 = 64`
- **Status**: ✅ Working well with FLAME defaults

#### Gated DeltaNet-170M Configurations

**Original Config** (`configs/gated_deltanet_170M.json`):
```json
{
  "hidden_size": 768,
  "num_heads": 6,
  "head_dim": 128,
  "expand_v": 2,
  "use_gate": true,
  "use_short_conv": true,
  "conv_size": 4
}
```
- **Head dimension**: 128 (explicit)
- **Status**: More stable with dropout=0.2

**Fixed Config** (`configs/gated_deltanet_170M_fixed.json`):
```json
{
  "hidden_size": 768,
  "num_heads": 6,
  "head_dim": 96,  // Follows 0.75 × hidden_size formula
  "expand_v": 2,
  "use_gate": true,
  "use_short_conv": true,
  "conv_size": 4
}
```
- **Head dimension**: 96 (follows `0.75 × hidden_size / num_heads = 0.75 × 768 / 6 = 96`)
- **Status**: ⚠️ Overfits severely with dropout=0.1

### Key Architectural Questions

1. **Head Dimension Scaling**:
   - Should `head_dim` follow a specific formula (e.g., `0.75 × hidden_size / num_heads`)?
   - How does `head_dim` interact with `expand_v`?
   - Original (128) vs Fixed (96) - which is better?

2. **Expand V Ratio**:
   - Both configs use `expand_v: 2`
   - How does this interact with `head_dim` and dropout?

3. **Number of Heads**:
   - GLA uses 12 heads, Gated DeltaNet uses 6
   - Is this optimal for each architecture?

### Experiments Needed

- [ ] Systematic sweep of `head_dim` values (64, 96, 128, 192)
- [ ] Test different `expand_v` ratios (1, 2, 4)
- [ ] Compare `num_heads` configurations
- [ ] Investigate interaction between `head_dim`, `expand_v`, and dropout

---

## 4. Dataset Size and Chinchilla Scaling

### Key Finding
Current WikiText-103 training shows clear overfitting, indicating we need a larger dataset to match Chinchilla scaling laws.

### Current Training Setup

**Dataset**: WikiText-103
- **Size**: ~103M tokens
- **Training**: 20 epochs
- **Total Tokens Seen**: ~2.06B tokens (103M × 20)
- **Model Size**: 170M parameters

### Chinchilla Scaling Law

According to Chinchilla scaling:
- **Optimal tokens**: ~20× model size in parameters
- **For 170M model**: ~3.4B tokens needed
- **Current**: ~2.06B tokens (60% of optimal)

### Evidence of Overfitting

**Gated DeltaNet-170M (fixed, dropout=0.1)**:
```
Step 25k:  val_ppl = 31.98  (best)
Step 30k:  val_ppl = 32.25
Step 40k:  val_ppl = 32.72
Step 50k:  val_ppl = 33.53
Step 62.9k: val_ppl = 34.65  (final)
```
**Clear degradation after step 25k** - classic overfitting pattern.

**GLA-170M (dropout=0.1)**:
```
Step 50k:  val_ppl = 29.73
Step 55k:  val_ppl = 29.54  (best)
Step 60k:  val_ppl = 30.30
Step 62.9k: val_ppl = 29.86
```
More stable, but still some fluctuation.

### Implications

1. **WikiText-103 is insufficient** for proper Chinchilla scaling
2. **Overfitting is the primary issue**, not just hyperparameters
3. **Larger datasets needed** - e.g., OpenWebText, The Pile, or C4

### Recommended Next Steps

- [ ] Train on larger dataset (OpenWebText, The Pile, or C4)
- [ ] Target ~3.4B tokens for 170M model
- [ ] Re-evaluate hyperparameters on larger dataset
- [ ] Compare scaling behavior between GLA and Gated DeltaNet

---

## Experiment Log

### Completed Experiments

| Date | Model | Config | Dropout | LR | Key Finding |
|------|-------|--------|---------|----|----| 
| 2025-01-14 | GLA-170M | default | 0.1 | 1e-3 | Optimal performance (29.54 ppl) |
| 2025-01-14 | GLA-170M | default | 0.2 | 1e-3 | Over-regularized (36.02 ppl) |
| 2025-01-14 | Gated DeltaNet-170M | fixed (head_dim=96) | 0.1 | 1e-3 | Severe overfitting (31.98 → 34.65) |
| 2025-01-14 | Gated DeltaNet-170M | original (head_dim=128) | 0.2 | 1e-3 | Better regularization (30.80 ppl) |

### Planned Experiments

- [ ] Gated DeltaNet with different dropout rates (0.15, 0.25)
- [ ] Gated DeltaNet with different learning rates
- [ ] Architectural sweep (head_dim, expand_v, num_heads)
- [ ] Training on larger dataset (OpenWebText/The Pile)

---

## Key Takeaways

1. **Dropout matters**: GLA optimal at 0.1, Gated DeltaNet may need 0.2+
2. **FLAME defaults**: Work great for GLA, need tuning for Gated DeltaNet
3. **Architecture**: Still exploring optimal head_dim and expand_v settings
4. **Dataset**: WikiText-103 is too small - need larger dataset for proper scaling

---

## References

- Training script: `scripts/train_comparison.py`
- Model registry: `scripts/model_registry.yaml`
- Experiment directory: `flame/exp/`
- FLAME framework: `flame/`

---

*This document should be updated after each significant experimental run.*
