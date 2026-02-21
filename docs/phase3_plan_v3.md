# Phase 3 Plan v3 Addendum - Tail-Risk 안정화 보강

## 요약
1. 기존 v2는 유지하고, 아래 9개를 강제 반영한다.
2. `CVaR` 정의를 `tail_fraction`으로 고정하고 기본값은 `0.2`로 변경한다.
3. hard-negative pool은 `훈련용`으로만 사용하고, `게이트/dual-λ 추정`은 고정 stress eval set으로 분리한다.
4. `fixed_stress_eval`의 크기/seed/샘플링 규칙/stress 정의를 명시 고정한다.
5. dual-λ 업데이트에 EMA/주기/클램프/증분 클립을 기본 규칙으로 고정한다.
6. Phase 3a `V-only`의 `K anchor`는 warm-start에서 고정한다.
7. `delimiter-only`는 유지하되 질량 상한/패널티/타입별 질량 로깅을 기본 활성화한다.
8. delimiter 제약의 적용 시점(`post-topk-renorm`, `per-slot`)을 고정한다.
9. 성공 기준을 `Gate 성공`과 `탐색 마일스톤`으로 분리해 판정 혼동을 제거한다.
10. 탐색 마일스톤 적용 범위(플래그 + 최대 run 수)를 운영 규칙으로 고정한다.

## 인터페이스/스키마 변경
1. `phase3.loss.off.type = cvar` 유지.
2. `phase3.loss.off.cvar_tail_fraction = 0.2` 추가.
3. `phase3.loss.off.cvar_q` 제거 또는 deprecated 처리.
4. `phase3.loss.off.train_source = hard_pool`.
5. `phase3.loss.off.constraint_source = fixed_stress_eval`.
6. `phase3.train.lambda_off.dual.metric_source = fixed_stress_eval`.
7. `phase3.train.lambda_off.dual.ema_beta = 0.9`.
8. `phase3.train.lambda_off.dual.update_every = 5` (steps).
9. `phase3.train.lambda_off.dual.lambda_min = 0.0`.
10. `phase3.train.lambda_off.dual.lambda_max = 10.0`.
11. `phase3.train.lambda_off.dual.delta_clip = 0.1` (per update).
12. `phase3.eval.off_fixed_stress.size = 2048`.
13. `phase3.eval.off_fixed_stress.seed = 3407`.
14. `phase3.eval.off_fixed_stress.sampling = once_per_experiment`.
15. `phase3.eval.off_fixed_stress.definition = phase2_stress_compatible`.
16. `phase3.mix.mode = v_only`에서 `phase3.mix.k_anchor.policy = warm_start_fixed`.
17. `phase3.candidate.delimiter.mass_cap = 0.15`.
18. `phase3.candidate.delimiter.mass_penalty = 0.05`.
19. `phase3.candidate.delimiter.apply_stage = post_topk_renorm_per_slot`.
20. `phase3.reporting.atom_type_mass = true`.
21. `phase3.milestone.explore.enabled = true`.
22. `phase3.milestone.explore.flag = phase3_explore`.
23. `phase3.milestone.explore.max_runs = 5`.

## 구현 규칙 고정
1. CVaR는 "상위 `tail_fraction` 평균"으로 계산한다.
2. hard pool 샘플은 `L_off` 미니배치 구성에만 사용한다.
3. `p99_stress` 추정, Gate B 판정, dual-λ 위반량 계산은 `fixed_stress_eval`에서만 수행한다.
4. `fixed_stress_eval`은 run 시작 시 1회 샘플링하고(`seed` 고정), run 내내 재샘플링하지 않는다.
5. `fixed_stress_eval`의 stress 정의는 Phase 2와 동일하게 유지한다.
6. Phase 2 호환 stress 정의:
7. `stress_delta(u) = max_i KL(p_student(.|u, state_i) || p_base(.|u))`.
8. dual-λ 업데이트는 raw p99가 아닌 EMA p99를 사용한다.
9. `p99_ema_t = beta * p99_ema_(t-1) + (1 - beta) * p99_raw_t`.
10. dual-λ 업데이트는 `update_every` step마다만 수행한다.
11. `delta = clip(eta * (p99_ema - threshold), -delta_clip, +delta_clip)`.
12. `lambda_off = clip(lambda_off + delta, lambda_min, lambda_max)`.
13. K anchor는 슬롯별 warm-start 시점의 `anchor_chunk/layer/position`을 전 학습 단계에서 고정한다.
14. delimiter cap/penalty는 슬롯별 top-k renorm 이후 확률질량에 적용한다.
15. delimiter 제약은 `per-slot`로 먼저 적용하고, run-level에서는 타입별 총 질량을 로깅한다.

## 성공 기준 분리
1. Gate 성공 기준(출시/판정용):
2. `Gate A = pass` AND `Gate B = pass`.
3. 탐색 마일스톤(진행 판단용, Gate 대체 아님):
4. `ON 유지`와 `OFF tail 개선` 동시 충족.
5. `ON 유지`: `on_gain_drop <= 0.01` (Phase 2 대비).
6. `OFF tail 개선`: `off_delta_p99_stress`가 Phase 2 대비 `>= 20%` 감소.
7. 탐색 마일스톤은 `phase3_explore=true`이고 run index가 `max_runs` 이내인 경우에만 적용한다.
8. 탐색 마일스톤 충족 시에도 Gate 결과는 별도로 `pass/fail` 기록한다.

## 테스트 추가
1. `test_phase3_cvar_tail_fraction_semantics.py`.
2. `test_phase3_fixed_stress_eval_spec.py`.
3. `test_phase3_dual_lambda_uses_fixed_eval_only.py`.
4. `test_phase3_dual_lambda_ema_and_clip.py`.
5. `test_phase3_hard_pool_not_used_for_gate.py`.
6. `test_phase3_k_anchor_warmstart_fixed.py`.
7. `test_phase3_delimiter_mass_cap_penalty.py`.
8. `test_phase3_delimiter_apply_stage_post_topk.py`.
9. `test_phase3_success_gate_vs_milestone_rule.py`.

## 가정/기본값
1. `cvar_tail_fraction=0.2`.
2. `off_fixed_stress.size=2048`.
3. `off_fixed_stress.seed=3407`.
4. `lambda_off.dual.ema_beta=0.9`.
5. `lambda_off.dual.update_every=5`.
6. `lambda_off.dual.lambda_range=[0.0, 10.0]`.
7. `lambda_off.dual.delta_clip=0.1`.
8. `k_anchor.policy=warm_start_fixed`.
9. `injection_scope=top25`.
10. `mix.mode=v_only`.
