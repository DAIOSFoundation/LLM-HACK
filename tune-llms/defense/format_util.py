"""Gemma-4 (google/gemma-4-E4B-it) 방어 파인튜닝용 포맷/마스킹 유틸.

설계 근거: defense_tuning_design.md §1(공통 포맷), §3(사실 스팬 마스킹), §3.5(build_labels).

제공 기능
- render_prompt / render_full : Gemma-4 chat template 렌더(train == serve == reward 일치).
- parse_fact_tags / parse_skeleton : gold 응답에서 '사실 스팬'을 char span으로 추출.
    · 방법 B(태그)  : «fact»...«/fact»
    · 방법 A(스켈레톤): [DECISION]/[REASON]/[FACT]/[ACTION]  (FACT 절만 마스크)
- build_masked_example : 프롬프트 -100 + 응답 학습 + 사실 스팬 -100.
- assert_gemma4_format : 레거시 포맷(<start_of_turn>, ### Instruction, ChatML) 누수 회귀 검사.

토크나이저 계약(중요)
    이 모듈은 특정 라이브러리에 의존하지 않는다. 넘겨받는 `tokenizer`가
      · apply_chat_template(messages, tokenize=False, add_generation_prompt=bool) -> str
      · __call__(text, add_special_tokens=False, return_offsets_mapping=True)
            -> {"input_ids": [...], "offset_mapping": [(s,e), ...]}
    두 인터페이스만 만족하면 된다. 프로덕션에서는 반드시 **fast 토크나이저**
    (GemmaTokenizerFast, use_fast=True) — offset_mapping 이 필요하다.
    Gemma-4 role 매핑: system->system, user->user, assistant->model.

주의
    - 델리미터를 코드에 하드코딩하지 말 것. 렌더링은 apply_chat_template 로 일원화한다.
      (아래 상수/GEMMA4_CHAT_TEMPLATE_REF 는 테스트/문서용 참조일 뿐, 프로덕션 경로에는 쓰지 않는다.)
    - 프롬프트/응답 경계는 offset(특수토큰이 (0,0) 오프셋을 주는 케이스)에 의존하지 않고
      '프롬프트 단독 토큰화 길이(n_prompt)'로 분리한다. offset 은 응답 내부 '사실 스팬'
      식별에만 사용한다.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

# --- Gemma-4 특수토큰(참조; raw chat_template.jinja + tokenizer_config.json 기준) -----------
GEMMA4_SPECIAL = {
    "bos": "<bos>",
    "eos": "<eos>",
    "pad": "<pad>",
    "unk": "<unk>",
    "turn_open": "<|turn>",   # 턴 열기
    "turn_close": "<turn|>",  # 턴 닫기
    "think": "<|think|>",
    "channel_open": "<|channel>",
    "channel_close": "<channel|>",
}

# 사실 태그(방법 B) / 스켈레톤 태그(방법 A)
FACT_OPEN, FACT_CLOSE = "«fact»", "«/fact»"   # «fact» … «/fact»
SKELETON_TAGS = ["[DECISION]", "[REASON]", "[FACT]", "[ACTION]"]

# 학습/서빙에 절대 새면 안 되는 레거시 포맷 마커(회귀 검사)
LEGACY_MARKERS = [
    "<start_of_turn>", "<end_of_turn>",      # Gemma-2/3
    "### Instruction", "### Response", "### Input",  # 구 tune-llms
    "<|im_start|>", "<|im_end|>",            # ChatML
]

# 문서/테스트용 참조 템플릿(프로덕션은 tokenizer.apply_chat_template 사용).
GEMMA4_CHAT_TEMPLATE_REF = (
    "<bos>"
    "{for each turn}: <|turn>{role}\\n{content}<turn|>\\n"
    "{if add_generation_prompt}: <|turn>model\\n"
)

Span = Tuple[int, int]


# --------------------------------------------------------------------------------------
# 메시지 렌더링
# --------------------------------------------------------------------------------------
def render_messages(system: str, user: str, response: str | None = None) -> List[Dict[str, str]]:
    """Gemma-4 chat 메시지 리스트. system 은 네이티브 역할로 배치(첫 user 접합 관례 폐기)."""
    msgs: List[Dict[str, str]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    if response is not None:
        msgs.append({"role": "assistant", "content": response})
    return msgs


def render_prompt(tokenizer, system: str, user: str) -> str:
    """추론/보상에서 모델에 주는 프롬프트 문자열(…<|turn>model\\n 까지)."""
    return tokenizer.apply_chat_template(
        render_messages(system, user), tokenize=False, add_generation_prompt=True
    )


def render_full(tokenizer, system: str, user: str, response: str) -> str:
    """프롬프트 + assistant 응답까지 렌더(학습 타깃 시퀀스)."""
    return tokenizer.apply_chat_template(
        render_messages(system, user, response), tokenize=False, add_generation_prompt=False
    )


# --------------------------------------------------------------------------------------
# 사실 스팬 파싱 (clean_text, fact_char_spans) 반환 — char span 은 clean_text 기준
# --------------------------------------------------------------------------------------
def parse_fact_tags(marked: str) -> Tuple[str, List[Span]]:
    """방법 B: «fact»…«/fact» 태그 제거 + 감싼 구간을 사실 스팬으로 기록."""
    clean_parts: List[str] = []
    spans: List[Span] = []
    i, pos = 0, 0
    n = len(marked)
    while i < n:
        if marked.startswith(FACT_OPEN, i):
            i += len(FACT_OPEN)
            j = marked.find(FACT_CLOSE, i)
            end = j if j != -1 else n
            content = marked[i:end]
            start = pos
            clean_parts.append(content)
            pos += len(content)
            spans.append((start, pos))
            i = end + len(FACT_CLOSE) if j != -1 else n
        else:
            nxt = marked.find(FACT_OPEN, i)
            end = nxt if nxt != -1 else n
            chunk = marked[i:end]
            clean_parts.append(chunk)
            pos += len(chunk)
            i = end
    return "".join(clean_parts), spans


_SKEL_RE = re.compile(
    r"\[(DECISION|REASON|FACT|ACTION)\]\s*(.*?)(?=\s*\[(?:DECISION|REASON|FACT|ACTION)\]|\Z)",
    re.S,
)


def parse_skeleton(marked: str, joiner: str = " ") -> Tuple[str, List[Span]]:
    """방법 A: [DECISION]/[REASON]/[FACT]/[ACTION] 골격.

    태그 라벨은 제거(서빙엔 태그 없이), 각 절의 '내용'만 joiner 로 이어 clean_text 구성.
    [FACT] 절 내용만 사실 스팬으로 마스크. 나머지(결정/정책/툴 액션)는 학습.
    """
    clean_parts: List[str] = []
    spans: List[Span] = []
    pos = 0
    for m in _SKEL_RE.finditer(marked):
        tag, content = m.group(1), m.group(2).strip()
        if not content:
            continue
        if clean_parts:  # 절 사이 구분자
            clean_parts.append(joiner)
            pos += len(joiner)
        start = pos
        clean_parts.append(content)
        pos += len(content)
        if tag == "FACT":
            spans.append((start, pos))
    return "".join(clean_parts), spans


# --------------------------------------------------------------------------------------
# 마스킹 예제 빌드 (§3.5 build_labels 를 실체화)
# --------------------------------------------------------------------------------------
def _overlaps(tok: Span, spans: Sequence[Span]) -> bool:
    cs, ce = tok
    for s, e in spans:
        if cs < e and s < ce:  # 구간 겹침
            return True
    return False


def build_masked_example(
    tokenizer,
    system: str,
    user: str,
    response_clean: str,
    fact_char_spans: Sequence[Span] = (),
    max_len: int = 2048,
) -> Dict[str, List[int]]:
    """프롬프트 -100 + 응답 학습 + 사실 스팬 -100 라벨을 만든다.

    반환: {"input_ids", "labels", "attention_mask"} (모두 max_len 로 절단).
    fact_char_spans 는 response_clean 기준 char (start, end) 목록.
    """
    prompt_text = render_prompt(tokenizer, system, user)
    full_text = render_full(tokenizer, system, user, response_clean)
    if not full_text.startswith(prompt_text):
        raise ValueError(
            "prompt 가 full 시퀀스의 접두가 아님 — chat_template 접두 일관성 확인 필요."
        )

    # 프롬프트/응답 경계: 특수토큰 offset (0,0) 문제를 피하려 '프롬프트 단독 토큰 길이'로 분리.
    n_prompt = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])

    enc = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids: List[int] = list(enc["input_ids"])
    offsets: List[Span] = list(enc["offset_mapping"])

    # 사실 스팬을 full_text 좌표로 이동(응답은 prompt_text 길이 지점부터 시작).
    base = len(prompt_text)
    fact_full = [(s + base, e + base) for (s, e) in fact_char_spans]

    labels: List[int] = list(input_ids)
    for i in range(len(input_ids)):
        if i < n_prompt:
            labels[i] = -100                      # 프롬프트(공격/정책) 마스크
        else:
            cs, ce = offsets[i]
            if cs != ce and fact_full and _overlaps((cs, ce), fact_full):
                labels[i] = -100                  # 응답 내 '사실' 마스크
            # else: 결정/정책/툴포맷/종료토큰 → 학습(그대로)

    input_ids = input_ids[:max_len]
    labels = labels[:max_len]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


def build_from_marked(
    tokenizer,
    system: str,
    user: str,
    response_marked: str,
    method: str = "tags",
    max_len: int = 2048,
) -> Dict[str, List[int]]:
    """마킹된 gold 응답('tags'|'skeleton'|'none')을 파싱해 곧장 마스킹 예제로."""
    if method == "tags":
        clean, spans = parse_fact_tags(response_marked)
    elif method == "skeleton":
        clean, spans = parse_skeleton(response_marked)
    elif method == "none":
        clean, spans = response_marked, []
    else:
        raise ValueError(f"unknown method: {method!r}")
    return build_masked_example(tokenizer, system, user, clean, spans, max_len=max_len)


# --------------------------------------------------------------------------------------
# 회귀 검사
# --------------------------------------------------------------------------------------
def assert_gemma4_format(text: str) -> None:
    """렌더 결과가 Gemma-4 포맷이고 레거시 마커가 없는지 확인(누수 회귀 방지)."""
    for m in LEGACY_MARKERS:
        if m in text:
            raise AssertionError(f"레거시 포맷 마커 누수: {m!r}")
    if GEMMA4_SPECIAL["turn_open"] not in text or GEMMA4_SPECIAL["turn_close"] not in text:
        raise AssertionError("Gemma-4 턴 델리미터(<|turn>/<turn|>) 누락")
