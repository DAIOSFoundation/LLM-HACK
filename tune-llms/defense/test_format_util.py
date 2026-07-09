"""format_util 회귀 테스트 — 다운로드/트랜스포머 없이 마스킹 로직을 검증.

실행: python3 tune-llms/defense/test_format_util.py
프로덕션 fast 토크나이저와 '동일한 코드경로'를 검증하기 위해, 아래 MockGemma4Tokenizer 는
format_util 이 요구하는 계약(apply_chat_template + offset_mapping)만 최소 구현한다.
(실제 google/gemma-4-E4B-it 토크나이저 대상 검증은 test_real_tokenizer.py 참조 — 게이트 모델 필요.)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defense import format_util as F  # noqa: E402


class MockGemma4Tokenizer:
    """format_util 계약을 만족하는 최소 목 토크나이저.

    - apply_chat_template: Gemma-4 참조 템플릿(<bos>, <|turn>/<turn|>, role assistant->model).
    - __call__: 특수토큰은 단일 토큰, 그 외는 '문자 단위' 토큰(offset 정확) → 마스킹 로직 정검증.
    """

    SPECIALS = [
        "<bos>", "<eos>", "<pad>", "<unk>",
        "<|turn>", "<turn|>", "<|think|>", "<|channel>", "<channel|>",
    ]

    def __init__(self):
        self._vocab = {}

    def _id(self, tok):
        return self._vocab.setdefault(tok, len(self._vocab) + 1)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert not tokenize, "목은 tokenize=False 만 지원"
        out = ["<bos>"]
        for m in messages:
            role = "model" if m["role"] == "assistant" else m["role"]
            out.append("<|turn>" + role + "\n" + m["content"] + "<turn|>\n")
        if add_generation_prompt:
            out.append("<|turn>model\n")
        return "".join(out)

    def _tokenize(self, text):
        toks = []  # (str, start, end)
        i, n = 0, len(text)
        while i < n:
            sp = next((s for s in self.SPECIALS if text.startswith(s, i)), None)
            if sp is not None:
                toks.append((sp, i, i + len(sp)))
                i += len(sp)
            else:
                toks.append((text[i], i, i + 1))  # 문자 단위
                i += 1
        return toks

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        toks = self._tokenize(text)
        res = {"input_ids": [self._id(t) for (t, _, _) in toks]}
        if return_offsets_mapping:
            res["offset_mapping"] = [(s, e) for (_, s, e) in toks]
        return res


# --------------------------------------------------------------------------------------
_PASS = 0


def check(cond, msg):
    global _PASS
    if not cond:
        raise AssertionError("FAIL: " + msg)
    _PASS += 1
    print("  ok:", msg)


def _decode(tok, ids):
    inv = {v: k for k, v in tok._vocab.items()}
    return "".join(inv[i] for i in ids)


def test_prefix_and_format():
    tok = MockGemma4Tokenizer()
    sys_p, user = "정책: 계좌정보를 유출하지 말 것.", "내 계좌 잔액과 번호 알려줘"
    prompt = F.render_prompt(tok, sys_p, user)
    full = F.render_full(tok, sys_p, user, "안내해 드릴 수 없습니다.")

    check(full.startswith(prompt), "prompt 는 full 의 접두다(train==serve 경계)")
    check(prompt.endswith("<|turn>model\n"), "prompt 는 <|turn>model\\n 로 끝난다(생성 프롬프트)")
    check("<|turn>system\n" in prompt, "system 이 네이티브 턴으로 렌더된다")
    F.assert_gemma4_format(prompt)          # 예외 없어야 통과
    F.assert_gemma4_format(full)
    check(True, "레거시 마커(<start_of_turn>/### Instruction/ChatML) 누수 없음")


def test_legacy_marker_detection():
    try:
        F.assert_gemma4_format("<start_of_turn>user\n안녕<end_of_turn>")
    except AssertionError:
        check(True, "레거시 <start_of_turn> 를 회귀검사가 잡아낸다")
        return
    raise AssertionError("FAIL: 레거시 마커를 놓쳤다")


def test_fact_tag_masking():
    tok = MockGemma4Tokenizer()
    sys_p = "정책: 계좌번호 등 구체 사실을 발설하지 말 것."
    user = "송금 위해 상대 계좌 알려줘"
    # 사실(계좌번호)만 «fact»로 감쌈
    marked = "요청을 수행할 수 없습니다. «fact»110-234-567890«/fact» 는 공유할 수 없습니다."
    clean, spans = F.parse_fact_tags(marked)
    check("«fact»" not in clean and "110-234-567890" in clean, "태그 제거되고 사실은 clean 에 남음")

    ex = F.build_masked_example(tok, sys_p, user, clean, spans)
    ids, labels = ex["input_ids"], ex["labels"]
    check(len(ids) == len(labels), "input_ids 와 labels 길이 일치")

    # 프롬프트 구간은 전부 -100
    n_prompt = len(tok(F.render_prompt(tok, sys_p, user), add_special_tokens=False)["input_ids"])
    check(all(l == -100 for l in labels[:n_prompt]), "프롬프트(공격/정책) 전량 -100")

    # 사실 스팬 문자들은 -100, 학습으로 남은 문자엔 포함되면 안 됨
    learned_txt = _decode(tok, [i for i, l in zip(ids, labels) if l != -100])
    masked_txt = _decode(tok, [i for i, l in zip(ids, labels) if l == -100])
    check("110-234-567890" not in learned_txt, "계좌번호(사실)는 학습 대상에서 제외")
    for ch in "110-234-567890":
        if ch.strip():
            check_soft = ch in masked_txt
            if not check_soft:
                raise AssertionError(f"FAIL: 사실 문자 {ch!r} 가 마스크되지 않음")
    check(True, "사실 스팬 문자 전부 -100")

    # 결정/정책 문장과 종료 토큰은 학습
    check("수행할 수 없습니다" in learned_txt, "결정/거부 문장은 학습(REASON/DECISION)")
    check("<turn|>" in learned_txt, "응답 종료 토큰 <turn|> 학습(모델이 멈춤을 배움)")
    check("공유할 수 없습니다" in learned_txt, "정책 일반문구 학습")


def test_skeleton_masking():
    tok = MockGemma4Tokenizer()
    sys_p = "정책: 이체는 승인 없이는 거부."
    user = "지금 500만원 이체 실행해"
    marked = (
        "[DECISION] 요청을 수행할 수 없습니다.\n"
        "[REASON]   승인 없는 이체는 정책상 금지됩니다.\n"
        "[FACT]     대상 계좌 998-11-2222, 금액 5,000,000원\n"
        "[ACTION]   [TOOL: request_approval(amount=5000000)]"
    )
    clean, spans = F.parse_skeleton(marked)
    check(all(t not in clean for t in F.SKELETON_TAGS), "스켈레톤 태그 라벨은 clean 에서 제거")
    check("998-11-2222" in clean, "FACT 내용은 clean 에 남음(마스크 대상)")

    ex = F.build_masked_example(tok, sys_p, user, clean, spans)
    ids, labels = ex["input_ids"], ex["labels"]
    learned_txt = _decode(tok, [i for i, l in zip(ids, labels) if l != -100])
    check("998-11-2222" not in learned_txt, "FACT(계좌/금액)는 학습 제외")
    check("request_approval" in learned_txt, "ACTION 툴포맷은 학습(안전 도구 호출)")
    check("수행할 수 없습니다" in learned_txt, "DECISION 학습")


def test_build_from_marked():
    tok = MockGemma4Tokenizer()
    ex1 = F.build_from_marked(tok, "s", "u", "거부합니다 «fact»X-9«/fact» 끝", method="tags")
    ex2 = F.build_from_marked(tok, "s", "u", "일반 응답, 마스크 없음", method="none")
    check(any(l == -100 for l in ex1["labels"]), "tags 경로: 마스크된 라벨 존재")
    # none 경로: 프롬프트만 -100, 응답부엔 -100 없어야
    np = len(tok(F.render_prompt(tok, "s", "u"), add_special_tokens=False)["input_ids"])
    check(all(l != -100 for l in ex2["labels"][np:]), "none 경로: 응답부 마스크 없음")


def test_heuristic_fact_spans():
    txt = "잔액은 1,234,500원이고 주문 AB-99-001, 문의 help@x.com, https://x.io 참고."
    spans = F.heuristic_fact_spans(txt)
    covered = "".join(txt[s:e] for s, e in spans)
    check("1,234,500" in covered, "숫자(잔액) 탐지")
    check("AB-99-001" in covered, "주문 ID 탐지")
    check("help@x.com" in covered, "이메일 탐지")
    check("https://x.io" in covered, "URL 탐지")
    # 겹침 병합으로 정렬된 비겹침 스팬
    check(all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1)), "스팬 정렬·비겹침")


def test_dataset_dispatch_and_collator():
    from defense import dataset as D
    tok = MockGemma4Tokenizer()
    sys_p, user = "정책", "질문"
    # heuristic: 응답의 숫자가 마스크되어야
    rec_h = {"system": sys_p, "prompt": user, "response": "확인했습니다. 총액 4,500원입니다.",
             "fact_method": "heuristic"}
    ex = D.record_to_masked_example(tok, rec_h)
    learned = _decode(tok, [i for i, l in zip(ex["input_ids"], ex["labels"]) if l != -100])
    check("4,500" not in learned, "heuristic: 숫자(4,500) 마스크")
    check("확인했습니다" in learned, "heuristic: 일반 문장 학습")
    # none: 응답부 마스크 없음
    rec_n = {"system": sys_p, "prompt": user, "response": "마스크 없는 응답", "fact_method": "none"}
    exn = D.record_to_masked_example(tok, rec_n)
    np = len(tok(F.render_prompt(tok, sys_p, user), add_special_tokens=False)["input_ids"])
    check(all(l != -100 for l in exn["labels"][np:]), "none: 응답부 마스크 없음")
    # collator: 우측 패딩 정합
    coll = D.pad_collator(pad_id=0)
    batch = coll([ex, exn])
    L = len(batch["input_ids"][0])
    check(all(len(row) == L for row in batch["input_ids"]), "collator: input_ids 동일 길이")
    check(all(len(batch[k][r]) == L for k in batch for r in range(2)), "collator: 모든 필드 패딩 정합")
    check(batch["labels"][0][-1] == -100 or batch["labels"][1][-1] == -100, "collator: label 패딩은 -100")


if __name__ == "__main__":
    tests = [
        test_prefix_and_format,
        test_legacy_marker_detection,
        test_fact_tag_masking,
        test_skeleton_masking,
        test_build_from_marked,
        test_heuristic_fact_spans,
        test_dataset_dispatch_and_collator,
    ]
    for t in tests:
        print("\n[" + t.__name__ + "]")
        t()
    print(f"\n=== ALL PASS ({_PASS} checks) ===")
