"""티켓 #4(보강) — benign 도메인 시드 데이터 검증·병합·통계.

레드팀 DB엔 '공격'만 있어 benign(정상) 데이터가 없다(설계 §2.2). 과잉거부(over-refusal)를
막으려면 카테고리별 정상 태스크 + 정답 행동(ref_action)이 필요하다.

이 스크립트는 '생성'을 직접 하지 않고(생성은 Claude/강건모델·아래 CANONICAL_PROMPT 로 수행),
생성된 benign_seed.jsonl 을 **검증(스키마 + reward 정합) · 중복제거 · 통계** 한다.
create_defense_dataset.py 가 이 파일을 읽어 WARMUP benign / GRPO benign / PREF benign 으로 병합한다.

benign_seed 레코드 스키마
  {category, prompt, response, ref_action?, over_refusal?, fact_method: tags|none}

reward 정합(품질 게이트, defense_reward 재사용)
  - ref_action 있는 레코드: reward("benign", response, ref_action=ref_action) > 0 이어야(정답 툴 방출).
  - over_refusal 있으면: is_refusal(over_refusal) == True 이어야(과잉거부 예시로서 유효).

실행:
  python3 tune-llms/defense/gen_benign_domain.py --validate [--seed PATH]
  python3 tune-llms/defense/gen_benign_domain.py --merge extra.jsonl [--seed PATH]
  python3 tune-llms/defense/gen_benign_domain.py --print-prompt   # 생성용 프롬프트 출력
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from typing import Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
_TUNE = os.path.dirname(_HERE)
sys.path.insert(0, _TUNE)
from defense import defense_reward as R  # noqa: E402

DEFAULT_SEED = os.path.join(_TUNE, "data", "defense", "benign_seed.jsonl")
CATEGORIES = ["physical_robot", "ecommerce", "finance", "workflow", "sexual_safety"]

CANONICAL_PROMPT = """너는 방어 파인튜닝용 'benign(정상) 도메인 데이터' 생성기다. 목적은 과잉거부 방지.
- prompt: 명백히 정상적인 도메인 요청(공격/우회/유해 아님). 다양한 표현·언어.
- response: 실제로 도움되는 gold 답변. 툴 필요 시 마지막 줄 [TOOL: name(args)] / (피지컬) [MOTION: id].
  변동적 사실(가격·잔액·날짜·주문번호·고유명)은 «fact»…«/fact» 로 감싸고 fact_method="tags"; 없으면 "none".
- ref_action: 기대 정답 툴/모션 id(순수 정보성 답변이면 생략).
- over_refusal: 같은 요청을 정책 핑계로 '잘못 거부'하는 짧은 예(나쁜 예).
카테고리: physical_robot / ecommerce / finance / workflow / sexual_safety.
출력: {"category","prompt","response","ref_action?","over_refusal?","fact_method"} JSONL."""


def load_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: str, recs: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _hash(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8", "ignore")).hexdigest()


def validate(recs: List[Dict]) -> Dict:
    """스키마 + reward 정합 검사. 반환: 통계 dict, 그리고 유효 레코드만 반환."""
    ok, bad = [], []
    reasons = Counter()
    seen = set()
    for r in recs:
        cat = r.get("category")
        prompt = (r.get("prompt") or "").strip()
        resp = (r.get("response") or "").strip()
        if cat not in CATEGORIES:
            reasons["bad_category"] += 1; bad.append(r); continue
        if not prompt or not resp:
            reasons["empty"] += 1; bad.append(r); continue
        h = _hash(cat + "|" + prompt)
        if h in seen:
            reasons["dup"] += 1; bad.append(r); continue
        seen.add(h)
        ref = r.get("ref_action")
        if ref:
            if R.reward("benign", resp, ref_action=ref) <= 0:
                reasons["ref_action_not_emitted"] += 1; bad.append(r); continue
        over = r.get("over_refusal")
        if over and not R.is_refusal(over):
            reasons["over_refusal_not_refusal"] += 1; bad.append(r); continue
        ok.append(r)
    stats = {
        "total_in": len(recs),
        "valid": len(ok),
        "invalid": len(bad),
        "invalid_reasons": dict(reasons),
        "by_category": dict(Counter(r["category"] for r in ok)),
        "with_ref_action": sum(1 for r in ok if r.get("ref_action")),
        "with_over_refusal": sum(1 for r in ok if r.get("over_refusal")),
    }
    return {"stats": stats, "valid_records": ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default=DEFAULT_SEED)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--merge", metavar="EXTRA_JSONL", help="추가 jsonl 을 seed 에 병합(검증 후 유효분만)")
    ap.add_argument("--print-prompt", action="store_true")
    args = ap.parse_args()

    if args.print_prompt:
        print(CANONICAL_PROMPT)
        return 0

    if not os.path.exists(args.seed):
        print(f"seed 없음: {args.seed}\n생성 프롬프트: --print-prompt", file=sys.stderr)
        return 1

    recs = load_jsonl(args.seed)
    if args.merge:
        recs += load_jsonl(args.merge)

    res = validate(recs)
    print(json.dumps(res["stats"], ensure_ascii=False, indent=2))

    if args.merge:
        write_jsonl(args.seed, res["valid_records"])
        print(f"\n→ 병합·검증 완료: {len(res['valid_records'])}건 저장 {args.seed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
