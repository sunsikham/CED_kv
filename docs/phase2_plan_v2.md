# Phase 2 Plan v2 - Select-KV First Success (HF Strict)

## Summary
- Phase 2 목표는 Phase 1.5에서 검증된 HF cache 관로 위에서 Select-KV(span selection)를 실제로 동작시키고, ON 개선 + OFF 안정성을 하드 게이트로 통과하는 것이다.
- 본 문서는 RoPE/position 처리, prefix padding, candidate 정책 의미, teacher 방향성 검증, random 분산 완화를 decision-complete로 고정한다.

## Fixed Decisions
1. Backend: HF strict only (`allow_mock_fallback=false`)
2. Gate mode: Gate A required + Gate B required
3. Positioning mode: `compact`
4. Prefix padding mode: `null_pad`
5. Random baseline: 5 trials mean
6. KL direction: `KL(student || base)`
7. ON answer format: single symbol

## Public Interface Changes
1. CLI: `--phase 2` 지원
2. CLI 옵션:
   - `--phase2-model-id`
   - `--phase2-device`
   - `--phase2-max-on`
   - `--phase2-max-off`
   - `--phase2-prefix-len`
   - `--phase2-select-mode`
   - `--phase2-dtype`
   - `--phase2-strict-hf`
3. Config: `phase2` 블록 추가
4. New module: `src/cedkv_mvp/eval_phase2.py`
5. Optional module: `src/cedkv_mvp/metrics_phase2.py`

## Config Schema (`phase2`)
1. `seed`
2. `model.id`
3. `runtime.backend=hf`
4. `runtime.allow_mock_fallback=false`
5. `runtime.device`
6. `runtime.torch_dtype`
7. `eval.on_samples`
8. `eval.off_samples`
9. `eval.repro_runs`
10. `eval.n_random_trials=5`
11. `eval.decoding=greedy`
12. `prefix_len`
13. `prefix.padding_mode=null_pad|shrink_to_selected`
14. `injection.positioning_mode=compact|absolute`
15. `select.mode=attention_diversity_span|delimiter_span|random_span|causal_drop_span`
16. `select.span_len`
17. `select.span_budget`
18. `select.min_span_distance`
19. `select.candidate_policy=input_and_delimiter_only`
20. `select.center_policy=start|center`
21. `layer_scope.selection=top25`
22. `layer_scope.injection=all`
23. `thresholds.on_gain_min`
24. `thresholds.off_delta_p99_max`
25. `thresholds.delta_on_min`
26. `thresholds.rel_kl_to_teacher_max=1.0`
27. `thresholds.repro_acc_tol`
28. `reporting.topk_tokens`

## Core Definitions
1. Candidate policy 의미:
   - 후보 제한은 span의 `start`(또는 `center`) 토큰에만 적용한다.
   - span 내부 토큰은 contiguous 규칙으로 포함될 수 있다.
2. OFF Delta:
   - `Delta(u) = KL(p_student(.|u,s_e) || p_base(.|u))`
3. Teacher alignment metric:
   - `rel_kl_to_teacher = KL(student_select || teacher) / (KL(base || teacher) + 1e-6)`

## Implementation Steps
1. Add phase2 CLI and config resolver.
2. Build phase2 eval loop with ON/OFF sample generation reuse.
3. Score demo tokens from teacher `(demo + query)` attention.
4. Select contiguous spans with diversity constraints.
5. Build compact selected-demo prefill KV.
6. Apply null padding to match `prefix_len`.
7. Run Base / Teacher / Student(select) / Student(random x 5).
8. Compute metrics and gate decisions.
9. Write report and artifacts.

## Gate Rules
1. Gate A required:
   - teacher incremental consistency
   - null effect
   - nonzero effect
   - roundtrip relative KL
   - repro tolerance
2. Gate B required:
   - `on_gain = acc_select - acc_base >= on_gain_min`
   - `off_delta_p99_select <= off_delta_p99_max`
   - `delta_on_mean_select >= delta_on_min`
   - `rel_kl_to_teacher_select <= rel_kl_to_teacher_max`
   - `acc_select >= mean(acc_random_5_trials)`
3. Final pass:
   - `gateA_pass && gateB_pass`

## Artifacts and Reports
1. `outputs/{run_id}/report_phase2.json`
2. `outputs/{run_id}/metrics.jsonl`
3. `outputs/{run_id}/config.yaml`
4. `outputs/{run_id}/artifacts/phase2_selection.json`
5. `outputs/{run_id}/artifacts/phase2_prefix_meta.json`
6. `outputs/{run_id}/artifacts/hf_runtime_meta.json`

## Metrics
1. Record types: `run`, `sample`
2. Run metrics:
   - `phase2_on_acc_base`
   - `phase2_on_acc_teacher`
   - `phase2_on_acc_student_select`
   - `phase2_on_acc_student_random_mean`
   - `phase2_on_gain_select`
   - `phase2_off_delta_p99_select`
   - `phase2_delta_on_mean_select`
   - `phase2_rel_kl_to_teacher_select`
   - `phase2_gateA_pass`
   - `phase2_gateB_pass`
3. Sample metrics:
   - `phase2_delta_on_select`
   - `phase2_delta_off_select`
   - `phase2_answer_logprob_gain`
   - `phase2_selection_score_attention`
   - `phase2_selection_score_causal_drop` (if enabled)

## Tests
1. `tests/phase2/test_candidate_policy_center_restriction.py`
2. `tests/phase2/test_prefix_padding_null_pad.py`
3. `tests/phase2/test_rel_kl_to_teacher_gate.py`
4. `tests/phase2/test_random_trials_mean_comparison.py`
5. `tests/phase2/test_answer_logprob_gain_metric.py`
6. `tests/phase2/test_phase2_mock_logic_smoke.py`
7. `tests/phase2/test_phase2_hf_smoke_optin.py` (`RUN_HF_TESTS=1`)

## DoD
1. `python -m cedkv_mvp --phase 2 ...` 실행 가능
2. report에 Gate A/B 근거 수치 포함
3. positioning/padding mode가 config/report/artifact에 일치
4. random 5회 평균 비교로 Gate B 판정
5. teacher alignment 지표 포함
6. phase2 테스트 통과

## Assumptions and Defaults
1. 기본 모델: `Qwen/Qwen2.5-1.5B-Instruct`
2. 목적: 성능 최적화가 아니라 state 효과 존재 + OFF 안정성 검증
3. `causal_drop_span`은 초기에는 advisory 분석용으로 사용
