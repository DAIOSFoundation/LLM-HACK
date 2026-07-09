"""티켓 #4/#6 공용 — 학습 레코드 → 마스킹된 학습 예제 변환.

WARMUP 레코드({category, system, prompt, response, kind, fact_method})를 받아
fact_method 에 따라 사실 스팬을 뽑고 format_util.build_masked_example 로 라벨을 만든다.

fact_method 디스패치(설계 §3.4)
  tags      : «fact»…«/fact» 태그 파싱(방법 B) — 주로 생성 benign gold.
  skeleton  : [DECISION]/[REASON]/[FACT]/[ACTION] 파싱(방법 A) — 생성 거부 gold.
  heuristic : 정규식 사실 탐지(방법 C) — 증류 거부 gold(안전망).
  none      : 마스킹 없음(완성부만) — 사실이 없는 짧은 응답.

이 모듈은 라이브러리 비의존(넘겨받는 tokenizer 계약만). 학습 프레임워크에서
`map(lambda r: record_to_masked_example(tok, r))` 형태로 사용.
"""

from __future__ import annotations

import json
from typing import Callable, Dict, Iterator, List

from defense import format_util as F


def fact_spans_for(response: str, fact_method: str):
    """fact_method 에 따라 (clean_text, fact_char_spans) 반환."""
    if fact_method == "tags":
        return F.parse_fact_tags(response)
    if fact_method == "skeleton":
        return F.parse_skeleton(response)
    if fact_method == "heuristic":
        return response, F.heuristic_fact_spans(response)
    if fact_method in (None, "", "none"):
        return response, []
    raise ValueError(f"unknown fact_method: {fact_method!r}")


def record_to_masked_example(tokenizer, rec: Dict, max_len: int = 2048) -> Dict[str, List[int]]:
    """WARMUP 레코드 → {input_ids, labels, attention_mask}."""
    clean, spans = fact_spans_for(rec.get("response", ""), rec.get("fact_method", "none"))
    return F.build_masked_example(
        tokenizer,
        rec.get("system", ""),
        rec.get("prompt", ""),
        clean,
        spans,
        max_len=max_len,
    )


def load_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def masked_example_iter(tokenizer, path: str, max_len: int = 2048) -> Iterator[Dict[str, List[int]]]:
    for rec in load_jsonl(path):
        yield record_to_masked_example(tokenizer, rec, max_len)


def pad_collator(pad_id: int, label_pad: int = -100) -> Callable:
    """input_ids/labels/attention_mask 를 배치 내 최대 길이로 우측 패딩하는 콜레이터.

    학습 프레임워크(HF Trainer 등)의 data_collator 로 주입. 반환 함수는 list[dict]→dict[list].
    (텐서 변환은 프레임워크/torch 유무에 맡긴다 — 여기서는 파이썬 리스트로 반환.)
    """
    def collate(features: List[Dict[str, List[int]]]) -> Dict[str, List[List[int]]]:
        maxlen = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "labels": [], "attention_mask": []}
        for f in features:
            n = maxlen - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [pad_id] * n)
            batch["labels"].append(f["labels"] + [label_pad] * n)
            batch["attention_mask"].append(f["attention_mask"] + [0] * n)
        return batch

    return collate
