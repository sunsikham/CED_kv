# Phase 1.5 Plan (HF Gate A Hardening)

## Objective
Replace Phase 1 Gate A mock-distribution checks with real HF logits/KV checks while keeping Gate B advisory.

## Required Gate A checks
1. A0 Teacher incremental consistency
   - compare `teacher_full(demo+query)` vs `teacher_cached(demo->past then query)`
   - metric: `KL(teacher_cached || teacher_full)` at answer first-token position
2. A1 Null effect
   - `KL(student_null || base)` near zero for ON/OFF
3. A2 Nonzero effect
   - `KL(student_full_demo || base)` must be above calibrated threshold
4. A3 Round-trip relative KL
   - `KL(student_full_demo || teacher_full) / (KL(base || teacher_full) + 1e-6) <= alpha`

## Safety constraints
- Gate A full-demo path always uses all layers.
- Full-vocab logits KL in float32 with log_softmax.
- Strict fail supported: when `backend=hf` and `allow_mock_fallback=false`, runtime errors must fail the run.

## Logging requirements
- Save HF runtime metadata:
  - demo/query/past lengths
  - query position ids first/last
  - attention mask length/sum
  - torch/transformers versions

## Test policy
- Phase0/Phase1(mock) tests always run.
- HF-specific Phase1.5 tests run only when `RUN_HF_TESTS=1`.

