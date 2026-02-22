# Phase 3-3 Pre-Execution Plan

## 목표
- Phase 3 보강 3번(`off_constraint_source` 역할 고정)을 운영 재현성 기준으로 마무리한다.
- 우선순위는 `3 -> 1 -> 2`이며, 이 문서는 3번 실행 직전 고정 스펙이다.

## 결정 사항 (A~E 반영 완료)
- 운영 모드 정의를 명시한다:
  - `phase3.runtime.policy_mode: prod | debug`
  - 기본은 `prod`
  - `CI=true`이면 내부적으로 `prod` 강제
- `debug_allow_nonfixed_constraint_sources`는 `policy_mode=debug`에서만 유효
- warn-only 카운트는 optimization step이 아니라 eval tick 기준으로 집계
- warn 지표는 Gate B와 동일 축으로 고정:
  - `off_delta_p99_stress`만 사용 (`typical` 금지)
  - `off_delta_p99_max` 기준 사용
  - source는 `fixed_stress_eval`만 사용
- `off_train_source`는 운영에서도 `hard_pool` 허용 (훈련 안정화 목적)
  - 단, constraint/dual metric source에는 non-fixed 금지 (prod 기준)

## 구현 범위

### 1) 설정/정책 고정
- `src/cedkv_mvp/eval_phase3.py`
  - `Phase3Settings`에 아래 필드 추가:
    - `policy_mode`
    - `constraint_warn_enabled`
    - `constraint_warn_patience_eval_ticks`
    - `constraint_warn_margin`
    - `constraint_eval_every`
  - `resolve_phase3_settings()`에서:
    - `policy_mode` 유효성 검사
    - `CI=true` 시 `policy_mode=prod` 강제
    - `prod`에서 `off_constraint_source`/`dual_metric_source`를 `fixed_stress_eval`로 강제
    - `debug_allow_nonfixed_constraint_sources`는 `debug`에서만 허용

### 2) warn-only 로직 (eval tick 기준)
- `src/cedkv_mvp/eval_phase3.py`
  - warn 카운트 업데이트 시점을 eval tick으로 제한
  - `patience_eval_ticks` 연속 초과 시 warn trace 기록
  - run abort/stop은 하지 않음
  - 기록 항목:
    - tick index
    - `off_delta_p99_stress`
    - threshold/margin
    - consecutive count
    - trigger 여부

### 3) 리포트/아티팩트/메트릭 정합
- `src/cedkv_mvp/eval_phase3.py`
  - report 필드 추가:
    - `runtime.policy_mode`
    - `runtime.policy_mode_effective`
    - `runtime.ci_forced_prod`
    - `off_loss.constraint_source_policy`
    - `off_loss.train_source_policy_note`
    - `warnings.constraint_warn_summary`
  - run metrics 추가:
    - `phase3_constraint_source_is_fixed`
    - `phase3_dual_metric_source_is_fixed`
    - `phase3_constraint_warn_trigger_count`
    - `phase3_constraint_warn_max_consecutive_eval_ticks`
- `src/cedkv_mvp/cli.py`
  - `phase3_constraint_warn_trace.json` 저장 매핑 추가

### 4) config/doc 반영
- `configs/base.yaml`
  - `phase3.runtime.policy_mode: prod`
  - `phase3.runtime.debug_allow_nonfixed_constraint_sources: false`
  - `phase3.train.constraint_warn` 블록 추가:
    - `enabled: true`
    - `patience_eval_ticks: 3`
    - `margin: 0.0`
    - `eval_every: 1`
- 문서 업데이트:
  - `docs/phase3_plan_v3.md`
  - `docs/phase3_todo_plan.md`
  - `docs/phase3_change_log.md`

## 테스트 계획
- 신규 테스트:
  1. `test_phase3_policy_mode_prod_enforces_fixed_sources.py`
  2. `test_phase3_policy_mode_debug_allows_nonfixed.py`
  3. `test_phase3_train_source_hard_pool_allowed_in_prod.py`
  4. `test_phase3_constraint_warn_eval_tick_semantics.py`
  5. `test_phase3_constraint_warn_metric_contract.py`
  6. `test_phase3_constraint_warn_artifact_write.py`
- 회귀 테스트:
  - 기존 phase3 테스트 전체
  - 전체 테스트 스위트

## 수용 기준 (DoD)
- 운영(`prod` 또는 `CI=true`)에서 constraint/dual source가 fixed가 아니면 즉시 실패
- debug 모드에서만 non-fixed source 허용
- 운영에서 `off_train_source=hard_pool` 정상 허용
- warn-only는 eval tick 기준으로 동작하고 run 중단 없음
- warn 지표가 Gate B 축(`off_delta_p99_stress`, fixed source, 동일 threshold)과 일치
- 신규/기존 테스트 모두 통과

## 구현 순서
1. 설정 스키마/검증 (`policy_mode`, CI 강제, source 규칙)
2. warn-only eval tick 계측
3. report/run metrics/artifact 출력 연결
4. config 기본값 반영
5. 테스트 작성 및 회귀 실행
6. 문서 업데이트
