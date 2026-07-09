"""티켓 #4 — redteam.db → 방어 학습 3세트(WARMUP / PREF / GRPO).

설계 근거: defense_tuning_design.md §2(데이터 설계), §2.4(품질 필터).

입력 : sfass-redteam/data/redteam.db  (attempts 테이블; objective=시나리오 id, payload=공격, agent_text=응답, success=뚫림여부)
       sfass-redteam/src/scenarios.py (objective→category/severity/action_keywords, CATEGORY_TARGET_PROMPTS)
       (선택) benign_seed.jsonl        (gen_benign_domain.py 산출; benign 도메인 태스크+정답행동)
출력 : tune-llms/data/defense/{warmup,pref,grpo}.jsonl + stats.json

세트 스키마
  WARMUP : {category, system, prompt, response, kind: benign|refuse, risk, fact_method, source}
  PREF   : {category, system, prompt, chosen, rejected, kind: refuse_vs_breach|benign_vs_refusal}
  GRPO   : {category, system, prompt, track: attack|benign, action_keywords?, ref_action?, objective?, risk}

실행: python3 tune-llms/defense/create_defense_dataset.py [--db PATH] [--benign PATH] [--out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_TUNE = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_TUNE)
_SFASS_SRC = os.path.join(_ROOT, "sfass-redteam", "src")

sys.path.insert(0, _TUNE)
sys.path.insert(0, _SFASS_SRC)

from defense import defense_reward as R  # noqa: E402

# scenarios 는 sfass-redteam/src 에서 로드(objective→category 매핑·타깃 프롬프트).
import scenarios as S  # noqa: E402

DEFAULT_DB = os.path.join(_ROOT, "sfass-redteam", "data", "redteam.db")
DEFAULT_BENIGN = os.path.join(_TUNE, "data", "defense", "benign_seed.jsonl")
DEFAULT_OUT = os.path.join(_TUNE, "data", "defense")

# 품질 필터 파라미터(§2.4)
MIN_LEN, MAX_LEN = 6, 1500
CATEGORIES = ["physical_robot", "ecommerce", "finance", "workflow", "sexual_safety"]


# --------------------------------------------------------------------------------------
# scenarios 인덱스
# --------------------------------------------------------------------------------------
def _scenario_index() -> Dict[str, Dict[str, Any]]:
    idx = {}
    for o in S.objectives():
        idx[o["id"]] = o
    return idx


def _target_prompt(category: str) -> str:
    ctp = getattr(S, "CATEGORY_TARGET_PROMPTS", {})
    val = ctp.get(category)
    if callable(getattr(S, "category_target_prompt", None)):
        try:
            val = S.category_target_prompt(category) or val
        except Exception:
            pass
    return val if isinstance(val, str) else ""


# --------------------------------------------------------------------------------------
# payload 정리 (일부는 {"payload": "..."} JSON 래핑)
# --------------------------------------------------------------------------------------
def clean_payload(raw: Optional[str]) -> str:
    if not raw:
        return ""
    s = raw.strip()
    if s.startswith("{"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                for k in ("payload", "prompt", "message", "text"):
                    if isinstance(obj.get(k), str) and obj[k].strip():
                        return obj[k].strip()
        except Exception:
            pass
    return s


def _printable_ratio(text: str) -> float:
    if not text:
        return 0.0
    printable = sum(1 for c in text if c.isprintable() or c in "\n\t")
    return printable / len(text)


def good_refusal(agent_text: str) -> bool:
    """§2.4 품질 필터: 명확·간결한 거부만 gold 로 채택."""
    if not agent_text:
        return False
    t = agent_text.strip()
    if not (MIN_LEN <= len(t) <= MAX_LEN):
        return False
    if _printable_ratio(t) < 0.95:
        return False
    if not R.is_refusal(t):          # 거부 마커 있고 툴 미방출
        return False
    return True


def _hash(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update((p or "").encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# 메인 빌드
# --------------------------------------------------------------------------------------
def build(db_path: str, benign_path: str, out_dir: str) -> Dict[str, Any]:
    idx = _scenario_index()
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = list(con.execute("select * from attempts"))
    con.close()

    warmup: List[Dict] = []
    pref: List[Dict] = []
    grpo: List[Dict] = []

    seen_refuse = set()
    seen_breach = set()
    # 카테고리별 (refuse[], breach[]) — PREF 페어링용
    by_cat_refuse: Dict[str, List[Dict]] = defaultdict(list)
    by_cat_breach: Dict[str, List[Dict]] = defaultdict(list)

    dropped = Counter()
    for r in rows:
        objective = r["objective"]
        scen = idx.get(objective)
        category = scen["category"] if scen else None
        if category not in CATEGORIES:
            dropped["unknown_category"] += 1
            continue
        action_keywords = scen.get("action_keywords", []) if scen else []
        risk = r["risk"] or (scen.get("severity") if scen else None)
        system = _target_prompt(category)
        prompt = clean_payload(r["payload"])
        agent_text = (r["agent_text"] or "").strip()
        if not prompt:
            dropped["empty_prompt"] += 1
            continue

        # GRPO attack 프롬프트 풀(성공/실패 무관 — 공격 프롬프트 자체가 자원)
        grpo.append({
            "category": category, "system": system, "prompt": prompt,
            "track": "attack", "action_keywords": action_keywords,
            "objective": objective, "risk": risk,
        })

        if r["success"] == 1:
            # 뚫림 응답 → PREF rejected / GRPO 음성예시
            if not agent_text:
                dropped["breach_no_text"] += 1
                continue
            key = _hash(prompt, agent_text)
            if key in seen_breach:
                dropped["dup_breach"] += 1
                continue
            seen_breach.add(key)
            by_cat_breach[category].append({
                "category": category, "system": system, "prompt": prompt,
                "response": agent_text, "risk": risk, "objective": objective,
            })
        else:
            # 거부 응답 → gold 거부(품질 필터)
            if not good_refusal(agent_text):
                dropped["low_quality_refusal"] += 1
                continue
            key = _hash(prompt, agent_text)
            if key in seen_refuse:
                dropped["dup_refuse"] += 1
                continue
            seen_refuse.add(key)
            rec = {
                "category": category, "system": system, "prompt": prompt,
                "response": agent_text, "kind": "refuse", "risk": risk,
                "fact_method": "heuristic",  # 증류 거부 → 사후 휴리스틱 마스킹(설계 §3.3/§3.4)
                "source": r["source"], "defender": r["defender_backend"],
            }
            warmup.append(rec)
            by_cat_refuse[category].append(rec)

    # PREF: 카테고리 내 (거부 chosen > 뚫림 rejected) 페어
    for cat in CATEGORIES:
        refs, brs = by_cat_refuse[cat], by_cat_breach[cat]
        n = min(len(refs), len(brs))
        for i in range(n):
            pref.append({
                "category": cat, "system": refs[i]["system"],
                "prompt": brs[i]["prompt"],           # 공격 프롬프트
                "chosen": refs[i]["response"],        # 정책준수 거부
                "rejected": brs[i]["response"],       # 뚫림
                "kind": "refuse_vs_breach",
            })

    # benign_seed 병합(있으면): WARMUP benign + GRPO benign + PREF(benign>거부)
    benign_stats = {"loaded": 0}
    if os.path.exists(benign_path):
        benign = _load_jsonl(benign_path)
        benign_stats["loaded"] = len(benign)
        for b in benign:
            cat = b.get("category")
            if cat not in CATEGORIES:
                continue
            system = _target_prompt(cat)
            resp = b.get("response") or b.get("gold_response") or ""
            warmup.append({
                "category": cat, "system": system, "prompt": b["prompt"],
                "response": resp, "kind": "benign", "risk": "low",
                "fact_method": b.get("fact_method", "skeleton"),
                "ref_action": b.get("ref_action"),
            })
            grpo.append({
                "category": cat, "system": system, "prompt": b["prompt"],
                "track": "benign", "ref_action": b.get("ref_action"),
            })
            # PREF benign 쌍(정답행동 > 과잉거부) — over_refusal 이 실제 거부일 때만
            if resp and b.get("over_refusal") and R.is_refusal(b["over_refusal"]):
                pref.append({
                    "category": cat, "system": system, "prompt": b["prompt"],
                    "chosen": resp, "rejected": b["over_refusal"],
                    "kind": "benign_vs_refusal",
                })

    os.makedirs(out_dir, exist_ok=True)
    _write_jsonl(os.path.join(out_dir, "warmup.jsonl"), warmup)
    _write_jsonl(os.path.join(out_dir, "pref.jsonl"), pref)
    _write_jsonl(os.path.join(out_dir, "grpo.jsonl"), grpo)

    stats = _stats(warmup, pref, grpo, dropped, benign_stats)
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


def _stats(warmup, pref, grpo, dropped, benign_stats) -> Dict[str, Any]:
    def cat_counter(recs, key=None):
        if key:
            return dict(Counter(r["category"] for r in recs if r.get(key) == key))
        return dict(Counter(r["category"] for r in recs))

    warm_kind = Counter(r["kind"] for r in warmup)
    warm_refuse_cat = dict(Counter(r["category"] for r in warmup if r["kind"] == "refuse"))
    warm_benign_cat = dict(Counter(r["category"] for r in warmup if r["kind"] == "benign"))
    grpo_track = Counter(r["track"] for r in grpo)
    return {
        "warmup": {
            "total": len(warmup),
            "by_kind": dict(warm_kind),
            "refuse_by_category": warm_refuse_cat,
            "benign_by_category": warm_benign_cat,
            "benign_refuse_ratio": round(warm_kind.get("benign", 0) / max(1, warm_kind.get("refuse", 1)), 3),
        },
        "pref": {
            "total": len(pref),
            "by_kind": dict(Counter(r["kind"] for r in pref)),
            "by_category": dict(Counter(r["category"] for r in pref)),
        },
        "grpo": {
            "total": len(grpo),
            "by_track": dict(grpo_track),
            "attack_by_category": dict(Counter(r["category"] for r in grpo if r["track"] == "attack")),
        },
        "dropped": dict(dropped),
        "benign_seed": benign_stats,
        "targets_note": "설계 §2.3: WARMUP ~2000, benign:refuse≈1:1, 5카테고리 균등. "
                        "benign 부족 시 gen_benign_domain.py 로 보강. physical 편중은 비피지컬 벤치 추가수집으로.",
    }


def _load_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _write_jsonl(path: str, recs: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--benign", default=DEFAULT_BENIGN)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    if not os.path.exists(args.db):
        print(f"ERROR: db 없음: {args.db}", file=sys.stderr)
        return 1
    stats = build(args.db, args.benign, args.out)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n→ {args.out}/{{warmup,pref,grpo}}.jsonl + stats.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
