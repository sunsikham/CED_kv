from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any


class HFBackendUnavailable(RuntimeError):
    """Raised when HF backend cannot be created in the current environment."""


@dataclass(frozen=True)
class HFBackendInfo:
    model_id: str
    device: str
    available: bool
    reason: str


@dataclass(frozen=True)
class HFModelRuntime:
    model_id: str
    device: str
    torch_dtype: str
    tokenizer: Any
    model: Any
    torch: Any
    transformers_version: str
    torch_version: str


def hf_backend_info(model_id: str, device: str, torch_dtype: str = "auto") -> HFBackendInfo:
    try:
        import transformers  # type: ignore  # noqa: F401
        import torch  # type: ignore  # noqa: F401
    except Exception as exc:
        return HFBackendInfo(
            model_id=model_id,
            device=device,
            available=False,
            reason=f"HF dependencies unavailable: {exc}",
        )

    return HFBackendInfo(
        model_id=model_id,
        device=device,
        available=True,
        reason="ok",
    )


def load_hf_model(model_id: str, device: str, torch_dtype: str = "auto") -> HFModelRuntime:
    info = hf_backend_info(model_id=model_id, device=device, torch_dtype=torch_dtype)
    if not info.available:
        raise HFBackendUnavailable(info.reason)

    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    import transformers  # type: ignore

    resolved_device = _resolve_device(device=device, torch_module=torch)
    dtype_obj = _resolve_dtype(dtype=torch_dtype, torch_module=torch)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {}
    if dtype_obj is not None:
        model_kwargs["torch_dtype"] = dtype_obj
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if resolved_device == "cuda":
        model = model.cuda()
    model.eval()

    return HFModelRuntime(
        model_id=model_id,
        device=resolved_device,
        torch_dtype=torch_dtype,
        tokenizer=tokenizer,
        model=model,
        torch=torch,
        transformers_version=str(transformers.__version__),
        torch_version=str(torch.__version__),
    )


def encode_text(runtime: HFModelRuntime, text: str) -> dict[str, Any]:
    encoded = runtime.tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(runtime.model.device)
    attention_mask = encoded["attention_mask"].to(runtime.model.device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def forward_next_token_logits(
    runtime: HFModelRuntime,
    text: str,
    use_cache: bool = True,
) -> dict[str, Any]:
    encoded = encode_text(runtime=runtime, text=text)
    with runtime.torch.no_grad():
        outputs = runtime.model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            use_cache=use_cache,
            return_dict=True,
        )
    logits = outputs.logits[:, -1, :].to(runtime.torch.float32)
    return {
        "logits": logits,
        "past_key_values": outputs.past_key_values if use_cache else None,
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


def forward_next_token_logits_with_past(
    runtime: HFModelRuntime,
    query_text: str,
    past_key_values: Any,
    past_len: int,
) -> dict[str, Any]:
    encoded = encode_text(runtime=runtime, text=query_text)
    query_ids = encoded["input_ids"]
    query_mask = encoded["attention_mask"]
    query_len = int(query_ids.shape[1])

    combined_mask = runtime.torch.ones(
        (1, past_len + query_len),
        dtype=query_mask.dtype,
        device=query_mask.device,
    )
    position_ids = runtime.torch.arange(
        past_len,
        past_len + query_len,
        device=query_ids.device,
        dtype=query_ids.dtype,
    ).unsqueeze(0)
    cache_position = runtime.torch.arange(
        past_len,
        past_len + query_len,
        device=query_ids.device,
    )

    kwargs: dict[str, Any] = {
        "input_ids": query_ids,
        "attention_mask": combined_mask,
        "past_key_values": past_key_values,
        "use_cache": True,
        "return_dict": True,
    }

    signature = inspect.signature(runtime.model.forward)
    if "position_ids" in signature.parameters:
        kwargs["position_ids"] = position_ids
    if "cache_position" in signature.parameters:
        kwargs["cache_position"] = cache_position

    with runtime.torch.no_grad():
        outputs = runtime.model(**kwargs)
    logits = outputs.logits[:, -1, :].to(runtime.torch.float32)
    return {
        "logits": logits,
        "past_key_values": outputs.past_key_values,
        "query_len": query_len,
        "past_len": past_len,
        "position_first": int(position_ids[0, 0].item()) if query_len > 0 else past_len,
        "position_last": int(position_ids[0, -1].item()) if query_len > 0 else past_len,
        "attention_mask_len": int(combined_mask.shape[1]),
        "attention_mask_sum": int(combined_mask.sum().item()),
    }


def answer_first_token_id(runtime: HFModelRuntime, answer_text: str) -> int | None:
    token_ids = runtime.tokenizer(
        answer_text,
        add_special_tokens=False,
    )["input_ids"]
    if not token_ids:
        return None
    return int(token_ids[0])


def answer_first_token_candidate_ids(
    runtime: HFModelRuntime,
    answer_text: str,
) -> list[int]:
    answer = str(answer_text).strip()
    if not answer:
        return []
    variants = [
        answer,
        f" {answer}",
        f"\n{answer}",
    ]
    ordered: list[int] = []
    for text in variants:
        token_ids = runtime.tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]
        if not token_ids:
            continue
        first = int(token_ids[0])
        if first not in ordered:
            ordered.append(first)
    return ordered


def kl_from_logits(
    student_logits: Any,
    base_logits: Any,
) -> float:
    log_q = student_logits.detach().to("cpu").float().log_softmax(dim=-1)
    log_p = base_logits.detach().to("cpu").float().log_softmax(dim=-1)
    q = log_q.exp()
    kl = (q * (log_q - log_p)).sum(dim=-1).mean()
    return float(kl.item())


def relative_kl(student_teacher_kl: float, base_teacher_kl: float, eps: float = 1e-6) -> float:
    return float(student_teacher_kl / (base_teacher_kl + eps))


def _resolve_device(device: str, torch_module: Any) -> str:
    if device == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if device == "cuda" and not torch_module.cuda.is_available():
        raise HFBackendUnavailable("Requested CUDA but torch.cuda.is_available() is False")
    if device not in {"cpu", "cuda"}:
        raise HFBackendUnavailable(f"Unsupported device value: {device}")
    return device


def _resolve_dtype(dtype: str, torch_module: Any) -> Any | None:
    if dtype == "auto":
        return None
    mapping = {
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }
    if dtype not in mapping:
        raise HFBackendUnavailable(f"Unsupported torch dtype: {dtype}")
    return mapping[dtype]
