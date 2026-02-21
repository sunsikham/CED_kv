# Phase 2 디버깅 정리 (HF 실실행 기준)

## 목적
- Phase 2(Select-KV v0)에서 Gate B 실패 원인을 분리하고, ON 평가 불성립 문제를 먼저 해결한다.

## 기준 런
- 1차 런: `phase2_hf_probe`
- 2차 런(수정 반영 후): `phase2_hf_probe_v2`
- 모델: `Qwen/Qwen2.5-1.5B-Instruct`
- 실행 옵션: `--phase 2 --phase2-strict-hf --phase2-max-on 2 --phase2-max-off 2`

---

## 1차 상태 (`phase2_hf_probe`)

### 결과 요약
- Gate A: pass
- Gate B: fail
- 주요 수치:
  - `on_acc_base=0.0`
  - `on_acc_teacher=0.0`
  - `on_acc_student_select=0.0`
  - `on_gain_select=0.0` (기준 `>=0.02` 실패)
  - `off_delta_p99_select=18.2131` (기준 `<=0.05` 실패)
  - `answer_logprob_gain_mean=9.8254` (양수)

### 해석
- 분포 변화는 큰데(`answer_logprob_gain` 양수), ON exact-match가 전부 0인 상태.
- 즉 선택 품질 이전에 ON 판정 정의/토크나이징 정합성이 깨져 있을 가능성이 큼.

---

## 디버깅 가설

1. ON 정답 판정을 단일 token id로 두면 공백/개행 토큰 차이에서 teacher도 0점이 될 수 있다.
2. 연속 지표(`answer_logprob_gain`)도 단일 id 기준이면 신뢰도가 낮다.
3. `Answer:` 포맷 불일치가 답 시작 토큰 위치를 흔들 수 있다.
4. OFF는 특정 state 하나 고정 주입 방식이라 worst-case steering이 과대 반영될 수 있다.

---

## 반영한 변경

### A. ON 판정 및 연속 지표 보강
- `src/cedkv_mvp/model_hf.py`
  - `answer_first_token_candidate_ids(...)` 추가
  - 후보 변형: `answer`, `" " + answer`, `"\n" + answer`
- `src/cedkv_mvp/eval_phase2.py`
  - `_answer_hit`를 `argmax in candidate_ids`로 변경
  - `_answer_logprob_gain`를 candidate 집합 `logsumexp` 기반으로 변경

### B. 프롬프트/정답 포맷 정합화
- `src/cedkv_mvp/synthetic.py`
  - `Answer:` -> `Answer: ` 로 통일
  - ON answer 문자열 `strip()` 정규화

### C. OFF 지표 분리(stress vs typical)
- `src/cedkv_mvp/eval_phase2.py`
  - `OFF-stress`: `max_i KL(student(state_i)||base)` per sample
  - `OFF-typical`: round-robin state 할당 후 KL
  - run metric 추가:
    - `phase2_off_delta_p99_stress`
    - `phase2_off_delta_p99_typical`
    - `phase2_off_delta_mean_stress`
    - `phase2_off_delta_mean_typical`
  - artifact 추가:
    - `outputs/{run_id}/artifacts/phase2/phase2_off_stress_top.json`
- `src/cedkv_mvp/cli.py`
  - phase2 artifact 저장에 `phase2_off_stress_top` 포함

### D. Teacher ON sanity gate 추가
- `src/cedkv_mvp/eval_phase2.py`
  - `on_acc_teacher >= teacher_min_on_acc` 미달 시
    - Gate B 즉시 fail
    - `failure_reason="on_eval_invalid"`
- `configs/base.yaml`
  - `phase2.thresholds.teacher_min_on_acc: 0.2` 추가

---

## 2차 상태 (`phase2_hf_probe_v2`)

### 결과 요약
- Gate A: pass
- Gate B: fail
- `failure_reason`: `off_delta_stress_p99_above_max`

### 핵심 수치 변화
- ON 측:
  - `on_acc_teacher: 0.0 -> 1.0`
  - `on_acc_student_select: 0.0 -> 1.0`
  - `on_gain_select: 0.0 -> 1.0`
  - `on_eval_valid: true` (teacher sanity 통과)
- OFF 측:
  - `off_delta_p99_stress: 18.2131` (기준 0.05 대비 매우 큼)
  - `off_delta_p99_typical: 18.2131` (이번 샘플에서는 stress와 동일)
  - `off_delta_mean_stress: 16.6832`
  - `off_delta_mean_typical: 16.6832`

### 결론
- ON 평가 불성립 문제는 해결됨.
- 현재 병목은 OFF 무해성(특히 stress/typical 모두 큰 steering)으로 명확히 분리됨.

---

## 생성된 산출물
- 1차 리포트: `outputs/phase2_hf_probe/report_phase2.json`
- 2차 리포트: `outputs/phase2_hf_probe_v2/report_phase2.json`
- 2차 OFF stress artifact:
  - `outputs/phase2_hf_probe_v2/artifacts/phase2/phase2_off_stress_top.json`

---

## 현재 상태 (요약)
- 완료:
  - ON 판정/지표 안정화
  - Teacher ON sanity gate
  - OFF stress/typical 분해 계측
  - `sdpa + output_attentions` 경고 제거(`eager` 일시 전환)
- 미해결:
  - OFF delta 기준 초과(게이트 실패 원인 확정)
- 다음 디버깅 포인트:
  - 선택 state의 OFF steering 완화(상태 필터링/게이팅/선택 정책 보정)
  - selection 스코어/후보 정책이 템플릿 토큰으로 쏠리는 문제 완화

---

## 추가 디버깅 (sdpa 경고 분리 + p99 상세 확인)

### E. `sdpa` attention 경고 대응
- 배경:
  - `output_attentions=True` 호출 시 `sdpa`에서는 경고가 발생했고 score 신뢰성에 의문이 있었다.
- 조치:
  - `src/cedkv_mvp/eval_phase2.py`의 `_attention_scores_query_to_prefill`에서
    - 스코어 계산 구간만 `model.set_attn_implementation(\"eager\")`로 일시 전환
    - 계산 후 기존 구현으로 복원
- 결과:
  - `sdpa/output_attentions` 경고는 사라짐.
  - 남은 경고는 `urllib3 NotOpenSSLWarning`만 확인됨.

### F. `urllib3 NotOpenSSLWarning` 의미
- 내용:
  - 현재 Python ssl이 LibreSSL 계열이고, urllib3 v2는 OpenSSL 1.1.1+를 권장한다는 경고.
- 영향:
  - HTTPS/환경 경고이며, Select-KV 로직/attention 스코어 계산과 직접적인 기능 오류는 아님.

### G. OFF p99 샘플 top-k 비교 결과
- stress 값(샘플 2개): `[2.3665, 11.2535]`, p99=`11.2535`
- p99 샘플 질의:
  - `Rewrite this sentence in a formal tone.`
- 비교:
  - base top-1: `\" \"` (prob `0.2886`)
  - student top-1: `\" sentence\"` (prob `0.2864`)
  - base가 student top-1(`\" sentence\"`)에 준 확률: `3.33e-05`
- 해석:
  - base에서 희박한 토큰을 student가 강하게 밀어 KL이 크게 커지는 패턴 확인.

### H. selected span 텍스트 덤프 결과
- 두 ON 에피소드에서 동일 span 반복 선택:
  - `[(0,4), (5,9), (10,14), (15,19)]`
- 복원된 compact text:
  - `Demo:\\nkey_ -> V0\\n_1 -> V\\nkey_2`
- 관찰:
  - `->` 등 템플릿/구조 토큰 비중이 높음.
  - 완전한 값 토큰(`V1`, `V2`) 직접 포함은 제한적.
  - 에피소드별 다양성 부족(초반 템플릿 구간 반복 선택).

### I. 최신 probe 결과 (`phase2_hf_attn_eager_check`)
- 실행: ON=1, OFF=1 (빠른 확인용)
- Gate A: pass
- Gate B: fail
- 실패 원인:
  - `failure_reason=off_delta_stress_p99_above_max`
  - `off_delta_p99_stress=10.2384` (기준 `0.05` 초과)
- 결론:
  - attention 경고 제거 이후에도 OFF 무해성 문제는 지속.
  - 즉 현재 병목은 경고 자체가 아니라 state/selection의 OFF steering 성질임.

---

## 추가 디버깅 2 (Phase 2.5 - no-template 후보 제외 AB)

### J. 목적
- delimiter/template 중심 후보를 제외하면 OFF 폭주가 줄어드는지 빠르게 검증.
- 같은 샘플(동일 seed)에서 baseline vs no-template 정책을 1회 비교.

### K. 실험 설정
- ON=2, OFF=4, seed=7
- 모델: `Qwen/Qwen2.5-1.5B-Instruct`
- 공통:
  - attention 기반 span 선택
  - compact prefix 주입
  - stress/typical OFF 계측
- 차이:
  - `baseline`: 기존 `input_and_delimiter_only`
  - `no_template`: 후보 중심점에서 `Demo/Question/Answer/:/->/개행/구분자` 계열 토큰 제외

### L. 결과 요약
- baseline:
  - `on_gain=1.0`
  - `off_stress_p99=11.5642`
  - `off_typical_p99=11.5642`
- no_template:
  - `on_gain=0.0`
  - `off_stress_p99=19.4373`
  - `off_typical_p99=19.4373`
- 변화:
  - `delta_off_stress_p99 = +7.8731`
  - `delta_off_typical_p99 = +7.8731`

### M. selection 덤프 관찰
- baseline compact text 예:
  - `Demo:\nkey_ -> V0\n -> V1\n -> V4\n\n`
- no_template compact text 예:
  - `0 -> V0key_1 ->2 -> V24 -> V4`
- 관찰:
  - no_template에서도 span 내부에 `->`, `V*`가 여전히 포함됨.
  - center 제한만 걸기 때문에 contiguous span 내부의 템플릿/값 토큰 유입은 막지 못함.

### N. 결론
- OFF 폭주는 “template 후보 선택 때문만”으로 설명되지 않음.
- 현 구조에서는 compact prefix 주입 자체가 강한 steering을 만들 가능성이 큼.
- 따라서 다음 단계는 단순 후보 필터링보다:
  - OFF-aware mixture 최적화(Phase 3의 `L_off`)와
  - prefix 효과 제어(게이팅/정규화/상태 강도 제약)가 우선이다.
