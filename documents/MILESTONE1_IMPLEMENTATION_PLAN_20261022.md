# Milestone 1 Implementation Plan

> Agent workers: execute with subagent-driven-development; task ownership and evidence below persist across compaction.

**Goal:** implement M1.1–M1.5 and the configurable E1–E10 matrix, then start only E1 base training, as authorized 2026-10-22.
**Architecture:** native PyTorch PixArt-alpha adaLN-single blocks; frozen 1024/frame latent contract; separate token-only condition projector and mT5 variants. Shared construction/checkpoint interfaces support training and generation.
**Tech Stack:** text2video conda, PyTorch2.6/cu124, transformers5.17, OmegaConf, existing I3D and frozen VAE.
**Spec:** MILESTONE1_ARCH_PIXART_20261022.md (Q1–Q9 resolved).

## Global Constraints
28 layers / 1152 width / 16 heads for real runs; c=t only; x0; frozen cache and VAE; E1=1a+4c+2a+3b+5b; ref layers 3/7/11/15/19/23/27. Text bridge Linear+LN trainable; 4c embedding+first3 actual encoder blocks trainable. Eval starts frame0,24 contiguous, skips shorter clips, GT lengths, shared test IDs. No pose/gloss/flow supervision. .trae never committed.

## Review Focus
Padding must not leak through either attention path; checkpoint/EMA must include fine-tuned text consistently; frame0 must be exactly once in generated and real clips; resume must restore optimizer and RNG/data position; single/DDP accumulated gradients must preserve effective batch and avoid duplicated samples.

## Execution rulings
Existing user-approved spec and explicit implement+train request authorize continuous execution; no repeated design approval. User explicitly requests switching current checkout, so operate on milestone1-pixart at c20c4a8 (not a new worktree). Preserve local migration files/frozen config materialization. Training budget defaults 30k optimizer updates, effective batch16 (revised to batch12 / 3 GPUs on 2026-10-22 per user GPU<=3 constraint; see TODO D65), lr1e-4, text_lr1e-5, bf16, EMA0.9999; motion lambda0.1 warmup2000 same E4/E9; DDIM50 CFG2 seed42. These are common recipe choices where spec leaves values open; record in configs. Only E1 launch; E2–E10 configuration/support only. No E10 weight download.

### Task 1: native PixArt and token condition bag
- Files: t2v/models/pixart.py, t2v/models/factory.py, tests/test_pixart.py.
- Interface: build_bundle(config,text_dim=768) -> bundle(model,condition_encoder); condition_encoder(text_embeddings,text_mask,ref_features,ref_drop_mask=None,lengths=None)->ConditionBag; model(x,t,cond,frame_mask)->[B,T,1024]. ConditionBag owns text_tokens/ref_tokens/text_mask/ref_mask and only token conditions.
- Tests first: actual 1a/1b/1c with 2a/2b/2c routing and all valid/padded batches, c independent of content, no cross params in1b, masked token perturbations invariant, gradient checkpoint gradients, pretrained block-only shape/key coverage with local synthetic weights.
- Implement official adaLN-single block semantics, native SDPA, temporal sincos, per-frame linear in/out, local block-only pretrain mapping and exact loaded/skipped key report; no network downloads.
- Verify focused suite; report red/green evidence.

### Task 2: real text depth variants
- Files: t2v/adapters/text.py, tests/test_milestone_text.py.
- Add optional num_layers argument, load mT5 then physically truncate encoder.block before moving to GPU (3 for4c,full for4a/4b), keep embedding trainability as declared; preserve existing adapter API/checkpoint restore. Encoder final norm explicit shared behavior.
- Tests first with tiny local model/checkpoint: executed layer count3, embedding and first3 gradients, no later blocks, 4a frozen, weights roundtrip, independent dropout/random eval state. No real optimizer updates on frozen files.

### Task 3: sampling and fixed24-frame evaluation
- Files: t2v/pipelines/milestone.py, tools/eval_milestone.py, tests/test_milestone_eval.py.
- Interfaces: load_pipeline(checkpoint,device,use_ema=False) uses build_bundle+TextEncoderAdapter(num_layers config); generate split samples uses GT T latent length then prepends real frame0 exactly once. checkpoint schema model,condition_encoder,text_encoder,ema={model,condition_encoder,text_encoder},config,step. Native VAE paths config.vae.{repo_path,checkpoint_path,config_path}; raw cached metadata relocated read-only to config.data.frame_root.
- Tests first: fixed eligible IDs = num_frames>=24, include frame0 exact, missing/short generations error, immutable protocol manifest/resume guard, stale output guard, same seed per ID independent processing order.
- CLI prepare ids/real floor and generate/evaluate; force include-frame0/contiguous/24 and min_frames24; keep real/generated identical sampling list. Call official tool, no metric reimplementation.

### Task 4: trainer/configuration and full integration (controller)
- Files: t2v/train.py, t2v/milestone_runtime.py, configs/milestone1/*.yaml, tests/test_milestone_training.py, tools/train_milestone.sh, README.md.
- Tests first: accumulation equivalence/optimizer updates, raw+completeEMA save/load, resume deterministic continuation, all10 configs factor matrix, condition mask consistency.
- Implement x0, bucket sampling, text separate lr, optional delta loss, gradient checkpointing; singleGPU and torchrun DDP effective-batch accumulation. No length model or global content vector. Atomic checkpoint includes CPU/CUDA/python/numpy RNG and batch cursor, text and all EMA modules; reject mismatched run configs, distributed singleton output lock.
- Add real portable data/VAE paths without changing frozen config/cache; log resolved settings and source identity. Small model overrides are testing only; production configs enforce spec dimensions.
- Verify baseline+focused tests, tiny actual training/generation path, official FVD24 selfcheck/floor on shared full eligible IDs. Update STRUCTURE and ledger.

### Task 5: review, commit, E1 launch
- Independent spec/code-quality review, fix findings and rerun relevant checks. Commit only owned implementation files; preserve existing frozen config type diff/untracked migration assets.
- Update TODO/FINISH/EXPERIMENT and record protocol IDs/floor; mark M1.6 ten-run results pending (not authorized to launch9other arms).
- Confirm GPU idle/disk; start E1 within the user GPU cap (<=3 GPUs since 2026-10-22, D65) with microbatch/accumulation meeting effective batch12; verify finite first optimizer updates, logs, process liveness, checkpoint and visualizer as feasible. Report actual run/log/PID/GPU and implementation commit; do not wait for30k completion.

## Progress
- Starting commit c20c4a8. Baseline tests pending target-branch result.
