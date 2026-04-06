# Parameter Golf — Config System

## Philosophy

Separate **what we tuned** from **what we invented**. Every parameter has provenance:
- **UPSTREAM** — OpenAI's `train_gpt.py` starter defaults
- **TUNED** — same parameter as upstream, different value (show both)
- **NOVEL** — parameter/technique that doesn't exist in upstream

## How to use

```bash
# Run with tuned baseline + active experiments:
source configs/tuned_baseline.env
source configs/exp_xsa.env
source configs/exp_bigram_hash.env
source configs/exp_value_embed.env
source configs/exp_ln_scale.env
source configs/exp_rope_dims.env
torchrun --nproc_per_node=8 train.py

# Run upstream baseline (for comparison):
source configs/upstream_baseline.env
torchrun --nproc_per_node=8 train_gpt.py
```

## Experiment Status

| Experiment | File | Status | Effect |
|------------|------|--------|--------|
| XSA all layers | `exp_xsa.env` | ACTIVE | XSA on all 11 layers |
| Bigram hash | `exp_bigram_hash.env` | BASELINE | 2048x128, now default (matches leaders) |
| Value embeddings | `exp_value_embed.env` | ACTIVE | Learned values in layers 9-10 |
| LN scale | `exp_ln_scale.env` | ACTIVE | Learnable LayerNorm scales |
| RoPE dims | `exp_rope_dims.env` | ACTIVE | Reduced to 16 dims |
| Bigram hash large | `exp_bigram_hash_large.env` | SHELVED | 3072x112 — leaders use 2048x128 |
| Progressive seq len | `exp_progressive_seq.env` | SHELVED | Not used by either leader |
| Test-Time Training | `exp_ttt.env` | SHELVED | #1 tried 25x, neutral/negative |
| Eval stride=16 | `exp_eval_stride.env` | SHELVED | Both leaders use stride=64 |
| Attention residuals | `exp_attn_res.env` | SHELVED | torch.compile breaks |
| Gated attention | `exp_gated_attention.env` | OFF | Not validated |
| Value residual | `exp_value_residual.env` | OFF | Not validated |
| QAT | `exp_qat.env` | OFF | Late quantization-aware |
| Multi-token pred | `exp_mtp.env` | OFF | Not validated |
| LAWA | `exp_lawa.env` | OFF | Alternative to SWA |

## Baseline Progression

### Upstream → Tuned (what we changed and why)

| Parameter | Upstream | Tuned | Why |
|-----------|----------|-------|-----|
| NUM_LAYERS | 9 | 11 | +2 layers fits 16MB budget |
| MLP_MULT | 2 | 3.0 | Wider MLPs, more capacity |
| TRAIN_SEQ_LEN | 1024 | 2048 | 2x context window |
| TRAIN_BATCH_TOKENS | 524288 | 786432 | 1.5x throughput |
| WARMDOWN_ITERS | 1200 | 4000 | Matches leader #1 |
| TIED_EMBED_LR | 0.05 | 0.035 | Tuned |
| MATRIX_LR | 0.04 | 0.025 | Tuned |
| SCALAR_LR | 0.04 | 0.025 | Tuned |
| MUON_MOMENTUM | 0.95 | 0.99 | Higher momentum for stability |
| MUON_MOMENTUM_WARMUP_START | 0.85 | 0.92 | Tuned |
| MUON_MOMENTUM_WARMUP_STEPS | 500 | 1500 | Longer warmup |
| GRAD_CLIP_NORM | 0.0 | 0.3 | Added gradient clipping |
| EVAL_STRIDE | — | 64 | Both leaders use 64 |
| BIGRAM_VOCAB_SIZE | — | 2048 | Both leaders use 2048 |
| BIGRAM_DIM | — | 128 | Both leaders use 128 |

### Current stack

train.py defaults = tuned_baseline + XSA + bigram_hash + value_embed + ln_scale + rope_dims + SWA + Full Hessian GPTQ + AR self-cal + ±1 pruning

## Results

| Config | bpb (4090) | bpb (8xH100) | Notes |
|--------|------------|--------------|-------|
| Upstream baseline | — | — | train_gpt.py defaults |
| Tuned baseline | ~1.41 | — | Fixed warmdown |
| train.py (prev) | 1.4073 | — | 413 steps, 10 min |
| train.py (aligned) | TBD | — | Aligned with leader params |
| Leaderboard #1 | — | 1.1147 | abaybektursun |
| Leaderboard #2 | — | 1.1194 | abaybektursun |
