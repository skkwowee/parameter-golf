# Parameter Golf — Experiment Notes

## Attention Residuals (AttnRes) — arxiv 2603.15031
**Status: Shelved — torch.compile incompatible**

Implemented Block AttnRes (softmax attention over previous layer block outputs instead of fixed residual accumulation). Results on 4090 with fixed warmdown:

| Step | Baseline | AttnRes |
|------|----------|---------|
| 100  | 1.9127   | 1.8325  |
| 200  | 1.6071   | 1.5808  |

AttnRes is ~0.03 bpb better per step, but 1.6x slower (2.15s vs 1.34s/step) due to torch.compile graph breaks. The `@torch.compiler.disable` on the aggregate function shatters the compiled graph into ~22 fragments. Tried:
- `fullgraph=True` on whole model — fails (dynamic list lengths)
- `fullgraph=True` on individual blocks — fails (varying ln_scale_factor constants)
- `fullgraph=False` on whole model — works but 1.6x overhead from fragmented graphs
- Eager mode (no compile) — ~10s/step, unusable
- Pre-allocated buffer with in-place writes — breaks autograd

The per-step improvement doesn't compensate for the speed penalty. On 8xH100 (~5600 vs 9000 steps), baseline would still win.

**Would need**: custom Triton kernel or torch.compile-friendly static-graph formulation to be viable.

### Update: Fixed-buffer approach compiles but is slow
Using `torch.where` for functional buffer writes makes `fullgraph=True` work, but `torch.where` copies the entire `[5, B, 2048, 512]` buffer ~15 times per forward = 150MB of memory copies = 120s/step (90x slower than baseline). The compile problem is solved but the memory bandwidth problem replaces it.

**Next approach to try**: Only build the buffer at block boundaries (3 times) instead of every layer. Within a block, accumulate partials normally and only use AttnRes attention at the block boundary transitions. This reduces buffer writes from 15 to 3.

## Warmdown Fix
The original wallclock-based warmdown (`lr_mul`) triggers from step 1 on slow GPUs (4090) because `warmdown_iters=3500` was tuned for 8xH100 (~9000 steps). Fixed by computing `warmdown_frac = warmdown_iters / iterations` and scaling proportionally to estimated total steps. This made baseline go from 1.7338 → 1.4073 bpb on 4090.

## Born-Again Distillation
**Status: Shelved — too expensive**

Snapshot model at warmdown start, use as teacher with KL loss. Extra forward pass doubles step time and VRAM (54GB). Not worth the step count reduction.

## Label Smoothing
**Status: Shelved — counterproductive**

Smoothing=0.05 deliberately reduces model confidence on correct predictions, which directly hurts bpb (the metric IS prediction confidence). Quantization-friendliness argument is weak when GPTQ already handles quantization error.

## Ideas Not Yet Tried
- **Int4 mixed quantization** — int4 for MLP weights, int6 for attention, more params in 16MB
- **Depth recurrence** — weight tying across layer groups, more effective depth for free
- **Progressive seq length** — start at 512, grow to 2048 (more docs per batch early)
- **Stride=1 sliding window eval** — current stride=64, pushing to 1 is free bpb at eval time
- **Full-param TTT** — current uses LoRA, full fine-tune on val data at eval (model is tiny)

## Baseline Numbers (4090, 10 min, fixed warmdown)
- train_top1.py, ATTN_RES=0: **1.4073 bpb** at 413 steps (1.34s/step)
- Previous V1 (broken warmdown): 1.5534 bpb at 436 steps
- Leaderboard #1 (8xH100): 1.1147 bpb
