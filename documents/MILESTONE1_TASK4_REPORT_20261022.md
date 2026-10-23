# Task4 initial implementation and validation

Trainer/config/runtime implementation committed; E1–E10 all configured. SingleGPU and torchrun DDP, bucketed effective batches, gradients accumulated with no_sync, text-specific LR, x0/motion, complete CPU EMA across backbone+projector+fine-tuned text. Atomic checkpoint includes optimizer, globalbatch cursor and perrank RNG; rejects incompatible experiment/world_size resume and existing outputs without resume; output lock oneowner.

RED: /tmp/m1-training-red.log missing t2v.milestone_runtime. GREEN: tests/test_milestone_training.py and tests/test_milestone_text.py =10passed. Includes actual 4step training vs2step+resume exact raw and EMA allcomponents and duplicateout rejection. Entire suite75passed/6warnings (before evaltask). Text task2 nits fixed: num_layers documentation and modified source weights forroundtrip.

Actual torchrun2rank GPU tiny-model smoke ran4 optimizersteps, accumulation2/effective4, produced config/logs/last/final; /tmp/m1-ddp-validation.log and runs/m1_ddp_validation_20261022. TinyGPU 200step run inprogress runs/m1_pipeline_validation_20261022 (small shape validation ONLY, notE1model and notcomparable scientific result). Callback imports future Task3 render_training_samples, disabled fortrainlifecycle tests; final integrationreview will verify callback.

Root recipe decisions commonall10:30kupdates/effective16/micro1/lr1e-4/text1e-5/fixedcosinex0/logitnormal/bf16/CPUEMA0.9999/DDIM50CFG2seed42/motion0.1warmup2000. E1 actualstillnotstarted. Metadata raw default avoids mismatchedEMA until complete EMA checkpoint selected.

## Review fixes
Both P2 fixed: motion objective reduced pervideo thenmicrobatchmean; initialization bypassed onresume with original savedmetadata restored. RED /tmp/m1-motion-red.log missing motion_loss_per_video; GREEN /tmp/m1-training-final.log11passed in5.84s, including deleted-initializer resume regression. DDP exact-continuation broadened: continuous2GPU6steps vs4steps+resume2steps all raw and EMA tensors forbackbone/projector/text exactlyequal. Logs /tmp/m1-ddp-continuous.log,/tmp/m1-ddp-resume.log.

## Motion padding follow-up
Reviewer correctly identified scale-count leakage from padded temporalextent. Regression changed to compare independently cropped true lengths; RED /tmp/m1-motion-trim-red.log reproduced failure4.1667vs5.5. Helper now trims each sample after its last valid frame before existingMultiScaleDeltaLoss. Fullsuite /tmp/m1-final-suite.log83passed,6warnings,9.47s.
