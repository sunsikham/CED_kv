# CED-KV MVP

Minimal bootstrap for CED-KV Phase 0-2 experimentation.

## Phase 0 quick start

```bash
python3 -m cedkv_mvp --config configs/base.yaml --phase 0 --stub-mode null
```

## Phase 1 quick start

```bash
python3 -m cedkv_mvp --config configs/base.yaml --phase 1
```

## Phase 1.5 HF quick start (strict fail)

```bash
python3 -m cedkv_mvp --config configs/base.yaml --phase 1 --phase1-backend hf --phase1-strict-hf
```

## Phase 3 quick start (HF strict)

```bash
python3 -m cedkv_mvp --config configs/base.yaml --phase 3 --phase3-strict-hf
```
