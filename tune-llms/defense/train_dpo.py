"""티켓 #6 (2/3) — 단계 2: DPO (오프라인 선호, 주력1).

설계: defense_tuning_design.md §5. (거부>뚫림) + (benign 정답>과잉거부) 선호를 KL 로 학습.

⚠️ GPU 필요 · API 검증 필요 (trl≥1.7 DPOTrainer, gemma-4). `# VERIFY:` 지점 박스에서 확인.

핵심 결정(설계 §5)
  - β=0.1, LR 5e-6, epoch 1~2. reference = 워밍업 어댑터. 정밀도 bf16.
  - 쌍: pref.jsonl 의 {prompt, chosen, rejected}. 프롬프트는 Gemma-4 템플릿으로 렌더(train==serve).
  - 텍스트 모방이 아니라 '결정'만 학습 + KL 로 능력 보존.

실행(박스): python3 tune-llms/defense/train_dpo.py --data data/defense/pref.jsonl \
                --warmup runs/warmup --out runs/dpo
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TUNE = os.path.dirname(_HERE)
sys.path.insert(0, _TUNE)

from defense import dataset as D
from defense import format_util as F

MODEL_ID = "unsloth/gemma-4-E4B-it"


def render_pref_row(tokenizer, r):
    """pref 레코드 → DPO 입력({prompt, chosen, rejected}) — 프롬프트를 Gemma-4 템플릿으로."""
    prompt = F.render_prompt(tokenizer, r.get("system", ""), r.get("prompt", ""))
    return {"prompt": prompt, "chosen": r["chosen"], "rejected": r["rejected"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(_TUNE, "data/defense/pref.jsonl"))
    ap.add_argument("--warmup", default=os.path.join(_TUNE, "runs/warmup"), help="워밍업 어댑터(정책·ref 초기값)")
    ap.add_argument("--out", default=os.path.join(_TUNE, "runs/dpo"))
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max_seq", type=int, default=2048)
    args = ap.parse_args()

    # 모델: 워밍업 어댑터에서 이어서(정책). ref 는 trl 이 정책 사본으로 관리(β KL 기준).
    from unsloth import FastModel  # noqa
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_ID, max_seq_length=args.max_seq, load_in_4bit=True, dtype=None,
    )
    # VERIFY: 워밍업 LoRA 어댑터 로드 방법(FastModel.get_peft_model + load_adapter, 또는 PeftModel.from_pretrained)
    model = FastModel.get_peft_model(model, r=16, lora_alpha=32, target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        finetune_vision_layers=False)
    if os.path.isdir(args.warmup):
        try:
            model.load_adapter(args.warmup, adapter_name="default")  # VERIFY
        except Exception as e:
            print(f"[warn] 워밍업 어댑터 로드 실패({e}) — base 에서 DPO 진행")

    from datasets import Dataset
    from trl import DPOTrainer, DPOConfig  # VERIFY: trl 1.7 심볼

    rows = [render_pref_row(tokenizer, r) for r in D.load_jsonl(args.data)]
    ds = Dataset.from_list(rows)

    cfg = DPOConfig(
        output_dir=args.out,
        beta=args.beta,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        bf16=True, fp16=False,
        gradient_checkpointing=True,
        max_length=args.max_seq,
        logging_steps=10,
        report_to=[],
    )
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tokenizer)  # VERIFY 인자명
    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"→ DPO 어댑터 저장: {args.out}")


if __name__ == "__main__":
    main()
