# Milestone 1 Task 1 implementation report

Status: implemented and committed as `da264fc` (`Add native PixArt latent backbone and token conditioning`). Owned changes only: `t2v/models/pixart.py`, `t2v/models/factory.py`, `tests/test_pixart.py`.

## Delivered

- Native PyTorch SDPA PixArt adaLN-single blocks; official block parameter names, shared timestep-only `t_block`, ungated cross residual, final timestep modulation table, zero final/cross output initialization.
- Frozen `[B,T,1024]` contract, per-frame linear input/output, temporal sin/cos, production defaults 28 layers / 1152 width / 16 heads, explicit `[text,text,text,ref]×7` schedule.
- `ConditionBag(text_tokens,ref_tokens,text_mask,ref_mask)` contains token conditions only. Trainable Linear→LayerNorm projections, reference dropout and empty text expose safe zero/null token. Masks use True=valid. Length argument is accepted and ignored.
- All cross/prefix/both modes: prefix-only allocates no cross modules; prefix tokens evolve through self-attention; both mode cross queries only video frames according to the layer plan. `none` routes allocate no cross module.
- Non-reentrant gradient checkpointing, checked against every parameter gradient in model and condition encoder.
- Strict local official state_dict / wrapped state_dict loading of blocks and shared t_block only, validates complete intended keys and shapes before mutation, exact sorted loaded/skipped key lists in `model.initialization_report`. No downloading.

## Verification evidence

RED: `OMP_NUM_THREADS=4 /home/wxy/anaconda3/envs/text2video/bin/python tests/test_pixart.py` failed before implementation with `AssertionError: native PixArt is missing`.

GREEN: same standalone command passes. `PYTHONPATH=. OMP_NUM_THREADS=4 /home/wxy/anaconda3/envs/text2video/bin/python -m pytest tests/test_pixart.py -q` reports **10 passed in 2.39s**. Coverage includes all nine injection/routing combinations, all-valid/padded frames, masked input invariance, valid text and reference sensitivity, safe unconditional tokens, independent timestep conditioning, gradient checkpoint equivalence, official residual equations, production dimensions, invalid contracts, and synthetic checkpoint loading/key rejection.

Full suite first attempt: `OMP_NUM_THREADS=4 .../python tests/run_all.py` passed all existing tests and PixArt but encountered concurrent work-in-progress tests `test_milestone_text.py` and `test_milestone_training.py` failing `ModuleNotFoundError: t2v`. Controller notified and repaired standalone entry points. Full-suite rerun `PYTHONPATH=. OMP_NUM_THREADS=4 /home/wxy/anaconda3/envs/text2video/bin/python tests/run_all.py` completed with **all tests passed**, exit 0. Legacy length-head test skips absent deployed head; text tests emit existing mT5→T5/extra `lm_head.weight` load warnings; text standalone pytest invocation emits an anyio assertion rewrite warning. No test failures remain in that run.

Owned-path `git diff --check` passed; commit contains exactly the three owned files. Pre-existing frozen-config type/whitespace change is outside this task and was not modified.

## Integration API

`build_bundle(config, text_dim=768)` returns `ModelBundle(model,condition_encoder,use_text=True,use_ref=True)`. Encoder accepts `(text_embeddings,text_mask,ref_features,ref_drop_mask=None,lengths=None)`. Model accepts `(x,t,cond,frame_mask=None)` and returns x0 `[B,T,1024]`, zeroing padded outputs. Reference feature width comes from `conditioning.reference.feature_dim` or `ref_dim`, defaults 512.

No network weight downloads, training jobs, frozen cache edits, or unrelated source edits were performed by this task. Official formula/name reference: https://github.com/PixArt-alpha/PixArt-alpha/tree/master/diffusion/model/nets.
