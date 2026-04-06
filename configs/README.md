# Parameter Golf — Config System

## Philosophy

Separate **what we tuned** from **what we invented**. Every parameter has provenance:
- **UPSTREAM** — OpenAI's `train_gpt.py` starter defaults
- **TUNED** — same parameter as upstream, different value (show both)
- **NOVEL** — parameter/technique that doesn't exist in upstream

## How to use

```bash
# Run with tuned baseline + specific experiments:
source configs/tuned_baseline.env
source configs/exp_xsa.env
source configs/exp_bigram_hash.env
source configs/exp_value_embed.env
source configs/exp_ln_scale.env
source configs/exp_rope_dims.env
torchrun --nproc_per_node=8 train_top2.py

# Run upstream baseline (for comparison):
source configs/upstream_baseline.env
torchrun --nproc_per_node=8 train_gpt.py
```

## Experiment Status

| Experiment | File | Status | In top1? | In top2? | Effect |
|------------|------|--------|----------|----------|--------|
| XSA all layers | `exp_xsa.env` | ACTIVE | Yes | Yes | XSA on all 11 layers |
| Bigram hash | `exp_bigram_hash.env` | ACTIVE | Yes | Yes | 2048-vocab bigram features |
| Bigram hash large | `exp_bigram_hash_large.env` | TESTING | No | Yes | 3072-vocab, dim=112 |
| Value embeddings | `exp_value_embed.env` | ACTIVE | Yes | Yes | Learned values in layers 9-10 |
| LN scale | `exp_ln_scale.env` | ACTIVE | Yes | Yes | Learnable LayerNorm scales |
| RoPE dims | `exp_rope_dims.env` | ACTIVE | Yes | Yes | Reduced to 16 dims |
| Progressive seq len | `exp_progressive_seq.env` | TESTING | No | Yes | 512→2048 ramp over 30% |
| Test-Time Training | `exp_ttt.env` | TESTING | No | Yes | Full-param adapt on val |
| Eval stride=16 | `exp_eval_stride.env` | TESTING | No | Yes | Denser sliding window |
| Attention residuals | `exp_attn_res.env` | SHELVED | No | No | torch.compile breaks |
| Gated attention | `exp_gated_attention.env` | OFF | No | No | Not validated |
| Value residual | `exp_value_residual.env` | OFF | No | No | Not validated |
| QAT | `exp_qat.env` | OFF | No | No | Late quantization-aware |
| Multi-token pred | `exp_mtp.env` | OFF | No | No | Not validated |
| LAWA | `exp_lawa.env` | OFF | No | No | Alternative to SWA |

## Baseline Progression

### Upstream → Tuned (what we changed and why)

| Parameter | Upstream | Tuned | Why |
|-----------|----------|-------|-----|
| NUM_LAYERS | 9 | 11 | +2 layers fits 16MB budget |
| MLP_MULT | 2 | 3.0 | Wider MLPs, more capacity |
| TRAIN_SEQ_LEN | 1024 | 2048 | 2x context window |
| TRAIN_BATCH_TOKENS | 524288 | 786432 | 1.5x throughput |
| WARMDOWN_ITERS | 1200 | 3500 | Proportional warmdown fix (NOTES.md) |
| TIED_EMBED_LR | 0.05 | 0.035 | Tuned |
| MATRIX_LR | 0.04 | 0.025 | Tuned |
| SCALAR_LR | 0.04 | 0.025 | Tuned |
| MUON_MOMENTUM | 0.95 | 0.99 | Higher momentum for stability |
| MUON_MOMENTUM_WARMUP_START | 0.85 | 0.92 | Tuned |
| MUON_MOMENTUM_WARMUP_STEPS | 500 | 1500 | Longer warmup |
| GRAD_CLIP_NORM | 0.0 | 0.3 | Added gradient clipping |

### Composition

- **top1** = tuned_baseline + XSA + bigram_hash + value_embed + ln_scale + rope_dims + SWA
- **top2** = top1 + progressive_seq + TTT + bigram_hash_large + eval_stride=16

## Results

| Config | bpb (4090) | bpb (8xH100) | Notes |
|--------|------------|--------------|-------|
| Upstream baseline | — | — | train_gpt.py defaults |
| Tuned baseline | ~1.41 | — | Fixed warmdown |
| top1 | 1.4073 | — | 413 steps, 10 min |
| top2 | TBD | — | Testing |
| Leaderboard #1 | — | 1.1147 | abaybektursun |
