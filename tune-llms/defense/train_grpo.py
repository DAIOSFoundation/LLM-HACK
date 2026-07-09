"""티켓 #6 (3/3) — 단계 3: GRPO (판정기 보상, 주력2·폐루프).

설계: defense_tuning_design.md §6. sfass 판정기의 '결정적 신호'(defense_reward)를 보상으로,
공격 성공률을 직접 최소화하면서 benign 도움성을 유지(과잉거부 방지).

⚠️ GPU 필수(vLLM MPS 미지원) · API 검증 필요 (trl≥1.7 GRPOTrainer/GRPOConfig, vllm≥0.23, gemma-4).
  `# VERIFY:` 지점 박스에서 확인. 보상 로직(defense.defense_reward)은 test_defense_reward.py 로 검증됨.

핵심 결정(설계 §6.1)
  - G=8, KL=0.04, LR=1e-6, temp=1.0, 공격:benign≈1:1.
  - 생성 백엔드 vLLM. 롤아웃마다 Claude 판정 금지 — 결정적 fast_judge 만.
  - 안정성(§6.4): train(HF)↔gen(vLLM) logprob 정합 — 양쪽 bf16, KL 유지, smoke-test 후 진입.

보상(§6.2)
  reward_fn 은 데이터셋 컬럼(track/action_keywords/ref_action)을 kwargs 로 받아
  defense_reward.reward(track, completion, action_keywords, ref_action) 를 그대로 호출.

실행(박스): python3 tune-llms/defense/train_grpo.py --data data/defense/grpo.jsonl \
                --init runs/dpo --out runs/grpo
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TUNE = os.path.dirname(_HERE)
sys.path.insert(0, _TUNE)

from defense import dataset as D
from defense import defense_reward as R
from defense import format_util as F

MODEL_ID = "unsloth/gemma-4-E4B-it"


def build_dataset(tokenizer, path):
    """grpo.jsonl → trl 데이터셋. prompt(Gemma-4 렌더) + 보상용 pass-through 컬럼."""
    from datasets import Dataset

    rows = []
    for r in D.load_jsonl(path):
        rows.append({
            "prompt": F.render_prompt(tokenizer, r.get("system", ""), r.get("prompt", "")),
            "track": r.get("track", "attack"),
            "action_keywords": r.get("action_keywords", []) or [],
            "ref_action": r.get("ref_action") or "",
        })
    return Dataset.from_list(rows)


def make_reward_fn():
    """trl GRPO reward_funcs: (completions, **columns) -> list[float]. (# VERIFY 시그니처)"""
    def reward_fn(completions, track=None, action_keywords=None, ref_action=None, **kw):
        n = len(completions)
        track = track or ["attack"] * n
        action_keywords = action_keywords or [[]] * n
        ref_action = ref_action or [""] * n
        out = []
        for i, comp in enumerate(completions):
            text = comp if isinstance(comp, str) else (comp[-1]["content"] if comp else "")
            out.append(R.reward(
                track[i], text,
                action_keywords=action_keywords[i] or [],
                ref_action=ref_action[i] or None,
            ))
        return out

    reward_fn.__name__ = "sfass_defense_reward"
    return reward_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(_TUNE, "data/defense/grpo.jsonl"))
    ap.add_argument("--init", default=os.path.join(_TUNE, "runs/dpo"), help="DPO 어댑터에서 이어서")
    ap.add_argument("--out", default=os.path.join(_TUNE, "runs/grpo"))
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--kl", type=float, default=0.04)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max_seq", type=int, default=2048)
    args = ap.parse_args()

    from unsloth import FastModel  # noqa
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_ID, max_seq_length=args.max_seq, load_in_4bit=True, dtype=None,
        fast_inference=True,   # vLLM 경로(# VERIFY: Unsloth+vLLM colocate 옵션)
    )
    model = FastModel.get_peft_model(model, r=16, lora_alpha=32, target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        finetune_vision_layers=False)
    if os.path.isdir(args.init):
        try:
            model.load_adapter(args.init, adapter_name="default")  # VERIFY
        except Exception as e:
            print(f"[warn] DPO 어댑터 로드 실패({e}) — 이전 단계 결과 없이 진행")

    ds = build_dataset(tokenizer, args.data)

    from trl import GRPOTrainer, GRPOConfig  # VERIFY: trl 1.7 심볼

    cfg = GRPOConfig(
        output_dir=args.out,
        num_generations=args.G,           # VERIFY 인자명(=G)
        beta=args.kl,                     # KL 계수(VERIFY: kl_coef vs beta)
        learning_rate=args.lr,
        temperature=args.temp,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_prompt_length=args.max_seq // 2,
        max_completion_length=512,
        bf16=True, fp16=False,
        use_vllm=True,                    # 온라인 생성(# VERIFY: vllm_mode/colocate)
        logging_steps=5,
        report_to=[],
    )
    trainer = GRPOTrainer(
        model=model, args=cfg, train_dataset=ds,
        reward_funcs=[make_reward_fn()],  # VERIFY: reward_funcs 인자명
        processing_class=tokenizer,
    )
    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"→ GRPO 어댑터 저장: {args.out}")


if __name__ == "__main__":
    main()
