# Phase 0 Plan (Implemented Baseline)

## Goal
Validate that the Phase 0 harness reacts correctly before real KV-prefix extraction is implemented.

## Scope
- ON metric: exact-match accuracy
- OFF metric: Delta KL
- Gate: `off_delta_p99 <= threshold`
- Outputs: `metrics.jsonl` and `report.json`

## Locked Definitions
- `stub_mode`: `null | global | selective`
- Delta direction: `KL(student || base)`
- Delta support: `union(top-k(student), top-k(base)) + OTHER`
- ON answer format (Phase 0): single symbol string (no whitespace)
- Sample schema includes `mode={on,off}`; predictors use `mode` field only
- Metrics `record_type`: `run | sample`

## Expected Stub Behavior
- `null`: student == base for all inputs
  - expected: OFF Delta near zero, gate pass
- `global`: student distribution shifts for both ON/OFF
  - expected: OFF Delta high, gate fail (exit code 2)
- `selective`: ON behaves like teacher, OFF behaves like base
  - expected: ON accuracy up, OFF Delta near zero, gate pass

## Key Artifacts
- `outputs/{run_id}/config.yaml`
- `outputs/{run_id}/metrics.jsonl`
- `outputs/{run_id}/report.json`

