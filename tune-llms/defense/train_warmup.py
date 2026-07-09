"""티켓 #6 (1/3) — 단계 1: 도메인 워밍업 (경량 Masked-SFT).

설계: defense_tuning_design.md §4. 베이스 google/gemma-4-E4B-it (텍스트 전용), 4-bit QLoRA.

⚠️ GPU 필요 · API 검증 필요
  이 스크립트는 CUDA + (transformers≥5.5.2 / trl≥1.7 / peft≥0.19 / unsloth) 환경에서만 실행된다.
  gemma-4 / Unsloth FastModel API 는 지식컷오프(2026-01) 이후 공개분이라, `# VERIFY:` 표시 지점은
  박스에서 실제 시그니처로 확인·조정할 것. 마스킹/데이터 로직(defense.dataset, format_util)은
  test_format_util.py 로 이미 검증됨 — 여기서는 '배선'만 남았다.

핵심 결정(설계 §4)
  - LoRA r16/α32, target = 언어 디코더 7모듈, PLE/비전/오디오 제외(all-linear 금지).
  - 정밀도 bf16(fp16 금지), grad-ckpt ON, LR 1e-4, 300~800 스텝(과학습 금지·early stop).
  - 로스 = 완성부 + 사실 스팬 마스킹(defense.dataset.record_to_masked_example).

실행(박스): python3 tune-llms/defense/train_warmup.py --data data/defense/warmup.jsonl --out runs/warmup
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TUNE = os.path.dirname(_HERE)
sys.path.insert(0, _TUNE)

from defense import dataset as D  # 검증됨(test_format_util.py)

MODEL_ID = "unsloth/gemma-4-E4B-it"  # 또는 google/gemma-4-E4B-it (+HF_TOKEN)
# 언어 디코더 한정 7모듈. PLE(embed_tokens_per_layer 등)·비전·오디오는 제외(설계 §4).
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_model_and_tokenizer(max_seq=2048):
    """Unsloth FastModel 로 4-bit 로드 + LoRA. (# VERIFY: 박스에서 시그니처 확인)"""
    from unsloth import FastModel  # noqa

    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=max_seq,
        load_in_4bit=True,          # NF4 QLoRA
        dtype=None,                 # bf16 자동(fp16 금지)
        full_finetuning=False,
    )
    model = FastModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=LORA_TARGETS,
        finetune_vision_layers=False,   # 비전 타워 동결
        finetune_language_layers=True,
        use_gradient_checkpointing="unsloth",
        # VERIFY: PLE/embed 제외가 자동인지 확인. 아니면 exclude_modules 로 명시 제외.
    )
    assert tokenizer.is_fast, "fast 토크나이저 필요(offset_mapping)"
    return model, tokenizer


class TorchMaskedCollator:
    """dataset.record_to_masked_example 결과를 우측 패딩 + torch 텐서화."""

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, features):
        import torch

        maxlen = max(len(f["input_ids"]) for f in features)
        ii, lb, am = [], [], []
        for f in features:
            n = maxlen - len(f["input_ids"])
            ii.append(f["input_ids"] + [self.pad_id] * n)
            lb.append(f["labels"] + [-100] * n)
            am.append(f["attention_mask"] + [0] * n)
        return {
            "input_ids": torch.tensor(ii, dtype=torch.long),
            "labels": torch.tensor(lb, dtype=torch.long),
            "attention_mask": torch.tensor(am, dtype=torch.long),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(_TUNE, "data/defense/warmup.jsonl"))
    ap.add_argument("--out", default=os.path.join(_TUNE, "runs/warmup"))
    ap.add_argument("--max_seq", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_steps", type=int, default=600)   # 300~800 (early stop)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    args = ap.parse_args()

    model, tokenizer = build_model_and_tokenizer(args.max_seq)

    # 레코드 → 마스킹 예제(검증된 디스패치). datasets 로 감싸 Trainer 에 전달.
    from datasets import Dataset

    recs = D.load_jsonl(args.data)
    examples = [D.record_to_masked_example(tokenizer, r, args.max_seq) for r in recs]
    ds = Dataset.from_list(examples)

    from transformers import Trainer, TrainingArguments

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        warmup_steps=min(50, args.max_steps // 10),
        bf16=True, fp16=False,                 # fp16 금지(설계 §4)
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=args.max_steps,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=TorchMaskedCollator(tokenizer.pad_token_id),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"→ warmup 어댑터 저장: {args.out}")


if __name__ == "__main__":
    main()
