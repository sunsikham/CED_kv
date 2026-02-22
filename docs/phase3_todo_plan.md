# Phase 3 To-Do Plan (Post-MVP -> Gate Pass)

## 목적
- 현재 상태: Phase 3 실행 MVP 완료 (`--phase 3` 실행, 리포트/아티팩트/기본 테스트 통과).
- 다음 목표: Gate B(`off_delta_p99_stress <= threshold`)를 안정적으로 통과하는 학습/튜닝 루프 완성.

## 현재 완료 범위
- Phase 3 CLI 경로 및 설정 스키마 추가.
- `fixed_stress_eval` 분리 및 고정 스펙 반영(크기/seed/정의).
- `L_off = CVaR(tail_fraction)` 기본 경로 반영.
- dual-λ 안정화 규칙 골격(EMA/update_every/clamp/delta_clip) 반영.
- delimiter mass cap/penalty + 적용 시점(`post_topk_renorm_per_slot`) 반영.
- Gate 성공 기준과 탐색 마일스톤 분리.

## 남은 핵심 작업 (우선순위 순)

### A. 보강 3개 (현재 리스크 기준, CPU 우선 가능, 우선순위: 3 -> 1 -> 2)

1. [~] (3번) `off_constraint_source` 역할 명확화 및 실행 경로 반영 강화
- 완료:
  - optimizer 루프에서 `off_constraint_source`/`dual_metric_source` 분기 반영.
  - `phase3_loss_off` 리포트가 `off_train_source`와 정합되도록 수정.
  - 관련 테스트 추가(`optimizer_source_routing`, `report_loss_off_source_alignment`).
  - `policy_mode(prod|debug)` + `CI=true -> prod` 강제 규칙 반영.
  - 운영 모드에서 non-fixed constraint/dual source 차단, debug override 경로 분리.
  - warn-only eval tick 추적 및 `phase3_constraint_warn_trace` 아티팩트 추가.
- 남음:
  - warn metric을 gate 축(`off_delta_p99_stress`)과 직접 결합한 고비용 검증 경로를 붙일지 결정.
  - early-stop/abort 기준을 실제로 도입할지(현재 warn-only) 최종 결정.

2. [ ] (1번) `layer_scope` / `mix_mode`를 실제 연산 경로에 연결
- 현재는 검증/리포트/메타 중심이라, 설정 변경 시 실제 학습 경로 변화가 제한적일 수 있음.
- 목표: `mixture_scope`, `injection_scope`, `mix_mode(v_only 등)`가 실제 forward/mix 계산 분기까지 영향을 주도록 코드 경로를 명시적으로 연결.

3. [ ] (2번) Proxy loss와 실제 Gate 지표 정합 강화
- 현재 `L_on/L_off`는 proxy 중심이라 Gate 지표(ON 품질, OFF p99)와 직접 정합이 약할 수 있음.
- 목표: step-wise로 proxy와 실제 지표(KL/정답 관련 지표/p99)의 상관을 로깅하고, 필요 시 일부 loss 항을 HF forward 기반 실측 지표로 치환/혼합.

### B. 기존 구현 항목 고도화

1. 학습 루프 실체화(최우선)
- 슬롯 mixture 파라미터(`w_{s,j}`)를 실제 optimizer로 업데이트.
- step 단위로 `top-k schedule (32->16->8->4)`가 실제 mixture 계산에 반영되도록 연결.
- `L = lambda_on * L_on + lambda_off * L_off + lambda_str * L_str`를 step-wise로 집계.

2. V-only mix 품질 고도화
- 현재 anchor 기반 조립 로직을 V-only 수식 관점으로 정밀화.
- `k_anchor.policy=warm_start_fixed`가 전 step에서 고정되는지 검증.
- layer scope(`mixture=top25`, `injection=top25`) 적용이 실제 forward 경로에 일관되게 반영되는지 확인.

3. OFF tail 안정화 튜닝
- dual-λ 하이퍼파라미터 튜닝:
  - `eta`, `ema_beta`, `update_every`, `delta_clip`, `lambda_max`
- 목표: λ 진동 최소화 + ON 유지 + OFF p99 하향.

4. hard pool vs fixed eval 분리 검증 강화
- 훈련 로스 샘플링은 hard pool만 사용.
- Gate/dual-λ 위반량 계산은 fixed stress eval만 사용.
- 코드 경로에 혼선이 없는지 통합 테스트 추가.

5. delimiter steering 회피 강화
- atom type mass 모니터링 대시보드/요약 추가.
- delimiter mass가 상한 근처로 몰릴 때 경고 로그 추가.
- 필요 시 cap 강화(예: 0.15 -> 0.10) 실험.

## 실험 계획 (권장 순서)

1. Smoke 검증
- 목적: 실행 경로/지표/아티팩트 정상 여부 확인.
- 설정 예시:
  - `on_samples=8`, `off_samples=32`, `off_fixed_stress.size=128`

2. Mid-scale 튜닝
- 목적: λ 안정화와 ON/OFF trade-off 탐색.
- 설정 예시:
  - `on_samples=32`, `off_samples=128`, `off_fixed_stress.size=512`

3. Full-scale 판정
- 목적: Gate 기준 충족 여부 최종 확인.
- 설정:
  - 기본 config 값 사용(`off_fixed_stress.size=2048`).

## 수용 기준 (DoD)
- Gate A pass.
- Gate B pass.
- 동일 설정으로 2회 재실행 시 Gate 결과 재현.
- 리포트에 아래 항목 포함:
  - `off_delta_p99_stress`, `off_delta_p99_typical`
  - `phase3_loss_on/off/str`
  - `lambda_off_trace`
  - atom type mass 요약

## 실행 커맨드 템플릿

```bash
python3 -m cedkv_mvp --config configs/base.yaml --phase 3 --phase3-strict-hf
```

## 리스크/주의
- full-scale은 시간/리소스 소모가 큼(특히 `off_fixed_stress`).
- 성급한 threshold 고정은 ON 붕괴를 유발할 수 있으므로, 먼저 mid-scale에서 λ 안정화부터 확인.
