# Preferred Route Implementation Plan

> Execution: implement inline using executing-plans and test-driven-development. User explicitly requested direct execution on 2026-10-13; proceed without another design/plan approval. Work in the current checkout, preserving existing user documentation edits.

Goal: enable pretrained mT5 full fine-tuning, then independently validate a text/video-latent matcher before adding its frozen semantic objective to x0 training.

Architecture: configurable text adapter with gradient support and checkpoint state; standalone train-split semantic pretraining with held-out eval retrieval gates; optional frozen semantic scorer taking predicted x0 (never conditioned DiT hidden). Existing frozen VAE, latent cache, normalizer and raw generation remain compatible.

Spec: in-chat approved route A→B and SIGN_VIDEO_LITERATURE_20261013.md.

Constraints: x0 output; 1024-d latent/frame; no pose/gloss/reference supervision; only train paired text/video for optimization; eval for selection; SignMbart only for final BT. Shared pretrained files remain read-only. No semantic benefit claimed without measured held-out results.

Review focus: legacy checkpoints; missing fine-tuned text weights; empty CFG embedding drift; parameter-group learning rates; semantic scorer leaking text or exploiting lengths; duplicate-text false negatives; validation mode/RNG; scorer checkpoint provenance and normalizer mismatch.

- [X] Task 1: tests for text gradient/update, fresh empty CFG, text state restoration and differential LR. Implement adapter/config/optimizer/checkpoint/generation/visualizer/dev mode wiring; retain old frozen path.
- [X] Task 2: tests for masked temporal semantics, duplicate-aware contrastive loss, held-out length-matched retrieval gate, frozen-scorer gradients into x0. Implement semantic model and train/eval entrypoint; save separate matcher checkpoint including its own fixed text reference and cache fingerprint.
- [X] Task 3: integrate validated scorer objective into training, using actual pred_x0 and original texts; reject unvalidated/incompatible scorer; add full-ft and semantic configs plus run sequence. Test live end-to-end gradient and reload.
- [X] Task 4: full suite, actual mT5 smoke and train→save→resume→inference validation passed; matcher trained/revalidated and semantic smoke completed. Formal A started; B/control/dev evaluation are scheduled, not yet completed.
- [X] Task 5: fresh review, important findings fixed with regression tests, commands/results recorded, STRUCTURE/DESIGN/NOTES updated. User TODO not expanded.

Ledger: initial GPU query unavailable in sandbox, outside sandbox GPUs 0/1/4 idle (~24GB); use GPU4 after fresh check. Python: /data/huyang/miniconda/envs/onvae/bin/python.

## Execution ledger (2026-10-13)

- Task 1: RED→GREEN fake-encoder gradient/update, fresh CFG, two learning rates and strict restore tests. Actual mT5 smoke 12 steps, resumed to 14; final norm changed by 0.00010222; restored text/CFG exact and deterministic inference passed (`reports/verification/text_finetuning.json`).
- Task 2: RED→GREEN temporal mask/order, duplicate-positive InfoNCE, frozen-teacher gradients and validation rejection. Full train matcher 40 epochs completed; selected epoch 10; stricter independent re-evaluation exported `validated_v2.pt` (128 dev R@1=.2422, 74 exactly length-matched videos pair accuracy=.9135, gap CI lower=.2459).
- Task 3: prediction x0 exposed with gradient, scorer freeze and cache fingerprint enforced, CFG dropped examples excluded. Actual semantic smoke 6 steps completed with checkpoints and comparison PNG/GIF/MP4 (`runs/_smoke_text_semantic_v2`). Initial GPU1 attempt interrupted by another process consuming memory; reran successfully on free GPU0 without interfering with that process.
- Task 4: fresh `/data/huyang/miniconda/envs/onvae/bin/python tests/run_all.py` → all 13 standalone test files passed. Formal A runs on GPU4, warm-start prefix60k; checkpoint and decode at step1000 passed. `runs/preferred_route/status.json` tracks persistent scheduler: B and equal-budget A-only continuation, correct/shuffled dev SignMbart, B-vs-A bootstrap, FVD and predicted-length deployment evaluation.
- Task 5: fresh independent reviewer found a flawed length baseline, ignored model source directory and lost restored frozen text state. Fixed with strict exactly equal-length controls/pure-length synthetic regression, anchored `/models/` and `/data/` ignore entries, and custom-weight checkpoint retention regression. Checkpoint interrupted-write regression also passed after atomic replacement fix.
- Ruling: preserve existing deployment until dev evidence supports replacement; generation improvement is not established by teacher retrieval alone. Keep user TODO entries unchanged; no commits/pushes or upstream cache writes.
- Deferred: generator has no automatic best-dev checkpoint selection (preexisting); current experiments use predeclared 5k endpoints. Near-synonym false negatives and clean-latent teacher transfer to generated latents remain empirical limitations, to assess using independent BT.
