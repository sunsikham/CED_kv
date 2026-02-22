# Phase 3 Change Log

## 문서 목적
- Phase 3 구현에서 실제 반영된 변경 사항을 코드/테스트 기준으로 빠르게 확인하기 위한 요약 로그.

## 최근 반영 커밋
- `(working)` phase3 step3 policy-mode/warn-only enforcement
- `7c55de6` phase3: align reported loss_off with off_train_source routing
- `bb2866d` phase3: enforce source routing, fix delimiter penalty path, and persist loss curve
- `6f07a1f` feat(phase3): add step-wise optimizer loop and top-k anneal wiring
- `a14237f` feat(phase3): add executable Phase 3 MVP pipeline

## 주요 변경 사항

### 1) 학습 루프/스케줄링
- step-wise optimizer 루프 추가 및 동작 연결.
- `top-k schedule (32->16->8->4)` 추적(`phase3_topk_schedule_trace`)과 step별 loss 추적(`phase3_loss_curve`) 반영.

### 2) OFF source 라우팅 정합
- optimizer 경로에서 `off_train_source`(`hard_pool` / `fixed_stress_eval`) 분기 반영.
- dual-λ 업데이트 metric source를 `dual_metric_source`로 선택하도록 반영.
- constraint p99 선택을 `off_constraint_source`로 반영.
- run report의 `phase3_loss_off`를 `off_train_source`와 일치하도록 수정.
  - 보조 지표도 함께 기록:
    - `phase3_loss_off_hard_pool_eval`
    - `phase3_loss_off_fixed_stress_eval`

### 3) delimiter 제약 안정화
- delimiter penalty가 중복 적용되지 않도록 정리.
- 최종 적용 시점은 `post_topk_renorm_per_slot` 경로에서만 적용되도록 통일.

### 4) 아티팩트 저장 보강
- `phase3_loss_curve`를 `artifacts/phase3/loss_curve.json`으로 저장하도록 CLI 매핑 추가.

### 5) Step3 정책 고정/경고 계측 (진행 중)
- `policy_mode(prod|debug)`와 `CI=true -> prod 강제` 정책 추가.
- 운영 모드에서 `off.constraint_source`/`dual.metric_source`를 `fixed_stress_eval`로 강제.
- `off_train_source=hard_pool`는 운영에서도 허용(훈련 경로 유지).
- warn-only 계측(`constraint_warn`)을 eval tick 기준으로 추적하고 별도 trace로 저장.
- `phase3_constraint_warn_trace` 아티팩트 저장 추가.

## 테스트 추가/보강
- `tests/phase3/test_phase3_optimizer_source_routing.py`
  - source 설정이 optimizer 경로에 반영되는지 검증.
- `tests/phase3/test_phase3_delimiter_penalty_single_apply.py`
  - delimiter penalty 단일 적용(중복 없음) 검증.
- `tests/phase3/test_phase3_loss_curve_artifact_write.py`
  - `phase3/loss_curve.json` 생성 검증.
- `tests/phase3/test_phase3_report_loss_off_source_alignment.py`
  - report `phase3_loss_off`와 `off_train_source` 정합 검증.

## 현재 상태 (3 -> 1 -> 2 기준)
- `[~]` 3번 `off_constraint_source` 역할 명확화: 주요 라우팅 반영 완료, 운영 정책 고정/early-stop 규칙 고정은 남음.
- `[ ]` 1번 `layer_scope/mix_mode` 실연산 연결: 미착수.
- `[ ]` 2번 proxy <-> 실제 Gate 지표 정합 강화: 미착수.

## 참고 문서
- 실행 TODO: `docs/phase3_todo_plan.md`
- 기준 플랜: `docs/phase3_plan_v3.md`
