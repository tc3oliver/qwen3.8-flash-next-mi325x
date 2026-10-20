[CK] [Bugfix] FMoE tuner: use all sources for shapes FlyDSL v2 cannot tune

## Summary

#5045 made `TUNE_ONLY` default to `flydslv2`. `gen_flydsl_v2_2stages_task` only covers bf16-out `per_1x32` g1u1 shapes, so with `TUNE_ONLY` unset every other shape gets 0 candidates:

    stage1 asm tasks is 0, tasks_ck is 0, task_1stage is 0
    no shape to tune or no solution found

Open config PRs work around this by setting `TUNE_ONLY=asm,cktile` (#5870, #6535).

With `TUNE_ONLY` unset, this PR keeps `flydslv2` for shapes it has candidates for and uses all sources (same as `TUNE_ONLY=`) for the rest. The check is per shape, so an untuned CSV that mixes MX and non-MX shapes works. Explicit `TUNE_ONLY` is unchanged. Reverting the default to all sources would undo #5045 for MXFP4 tuning.

Side effect: `OPUS_ONLY=1` without `TUNE_ONLY=` now applies to non-MX shapes. On main the non-empty default overrides it.

Non-MX shapes now reach the gfx942 asm FLAT candidates by default, the same as with `TUNE_ONLY=`; see #6013.

## Testing

- MI325X (gfx942), Qwen3.8-Flash-Next FP8 `per_1x128` MoE, 16 shapes, `TUNE_ONLY` unset: main gives 0 candidates per shape and exits 1 (aiter 0.1.24.post1; the source-selection code is unchanged on main). With this PR, `TUNE_ONLY` still unset (M=4, E=256): `stage1 asm tasks is 18, tasks_ck is 20, task_1stage is 6`, the same as `TUNE_ONLY=`. The shape tuned and the tuner exited 0.
- black / ruff clean.

Conflicts with #5825 (same line). Whichever lands second needs a trivial rebase.
