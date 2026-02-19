# Phase 1 Plan (Implemented)

## Objective
Validate KV extraction/injection pipeline viability before optimization-heavy phases.

## Gate Strategy
- Gate A (required): pipeline validity
  - teacher sanity (`teacher_acc >= 0.80`)
  - null prefix near-zero effect
  - non-null prefix non-zero ON effect
  - round-trip relative KL check at answer first token
- Gate B (advisory): contiguous span selection sanity

## Safety
- RoPE safety strategy: staged contiguous
  - Gate A uses all-layer full-demo round-trip
  - Gate B uses top25 contiguous spans
  - token-level non-contiguous injection deferred

## Artifacts
- `outputs/{run_id}/report_phase1.json`
- `outputs/{run_id}/artifacts/phase1/kv_capture_meta.json`
- `outputs/{run_id}/artifacts/phase1/selected_spans.json`

