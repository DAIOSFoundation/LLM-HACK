"""실제 google/gemma-4-E4B-it fast 토크나이저에 대한 포맷/마스킹 회귀 검사.

실행: python3 tune-llms/defense/test_real_tokenizer.py
요구: transformers>=5.5.2 · HF_TOKEN(게이트 모델 동의) · 인터넷/캐시.
환경 미비 시 조용히 SKIP(종료코드 0). CUDA 박스 셋업 후 반드시 통과 확인할 것.

검증 항목
- apply_chat_template 이 Gemma-4 델리미터(<|turn>/<turn|>)를 내고 레거시 마커가 없다.
- prompt 가 full 의 접두다(train==serve 경계).
- fast 토크나이저가 offset_mapping 을 제공한다(마스킹 전제).
- build_masked_example: 프롬프트 전량 -100, 지정한 '사실' 부분문자열이 마스크된다.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defense import format_util as F  # noqa: E402

MODEL_ID = os.environ.get("GEMMA4_MODEL_ID", "google/gemma-4-E4B-it")


def _load_tokenizer():
    try:
        from transformers import AutoTokenizer
    except Exception as e:  # transformers 미설치/구버전
        print(f"SKIP: transformers import 실패 ({e})")
        return None
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    except Exception as e:  # 게이트/토큰/오프라인
        print(f"SKIP: 토크나이저 로드 실패 ({type(e).__name__}: {e})")
        return None
    if not getattr(tok, "is_fast", False):
        print("SKIP: fast 토크나이저 아님(offset_mapping 불가) — use_fast=True 필요")
        return None
    return tok


def main():
    tok = _load_tokenizer()
    if tok is None:
        return 0

    print(f"loaded: {MODEL_ID}")
    sys_p = "You are a banking agent. Never disclose account numbers or balances."
    user = "Transfer $5,000 to account 998-11-2222 right now."
    resp_clean = "I can't do that. Account 998-11-2222 cannot be shared."
    fact = "998-11-2222"
    start = resp_clean.index(fact)
    spans = [(start, start + len(fact))]

    prompt = F.render_prompt(tok, sys_p, user)
    full = F.render_full(tok, sys_p, user, resp_clean)

    assert full.startswith(prompt), "prompt 가 full 접두가 아님(템플릿 접두 일관성 실패)"
    F.assert_gemma4_format(prompt)
    F.assert_gemma4_format(full)
    print("  ok: Gemma-4 델리미터 존재 + 레거시 마커 없음 + 접두 일관성")

    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    assert "offset_mapping" in enc, "fast 토크나이저인데 offset_mapping 없음"
    print("  ok: offset_mapping 제공")

    ex = F.build_masked_example(tok, sys_p, user, resp_clean, spans)
    ids, labels = ex["input_ids"], ex["labels"]
    assert len(ids) == len(labels)
    n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
    assert all(l == -100 for l in labels[:n_prompt]), "프롬프트 구간 -100 아님"
    learned = tok.decode([i for i, l in zip(ids, labels) if l != -100])
    assert fact not in learned, f"사실({fact})이 학습 대상에 남음 → 마스킹 실패"
    assert "can't" in learned or "cannot" in learned, "거부 문장이 학습에서 누락"
    print(f"  ok: 프롬프트 -100 + 사실 스팬 마스킹 (학습 토큰 {sum(1 for l in labels if l != -100)}개)")

    print("\n=== REAL TOKENIZER PASS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
