# defense/ — 방어 파인튜닝(gemma-4-E4B-it) 구현

설계: [../defense_tuning_design.md](../defense_tuning_design.md) · 플랜: [../defense_tuning_plan.md](../defense_tuning_plan.md)

베이스 **`google/gemma-4-E4B-it`**(텍스트 전용) · 3단계(워밍업 → DPO → GRPO) · **CUDA 전 구간**(Unsloth `FastModel` + 4bit QLoRA, GRPO=vLLM).

## 착수 티켓 진행

| # | 티켓 | 파일 | 상태 |
|---|---|---|---|
| 1 | env(CUDA) | (GPU 박스에서) | 대기 |
| **2** | **format util** — Gemma-4 template + 사실 스팬 마스킹 | `format_util.py` | **완료** |
| 3 | LoRA 주입 스모크 | (GPU) | 대기 |
| 4 | data 파이프라인 | `create_defense_dataset.py`(예정) | 대기 |
| 5 | reward | `defense_reward.py`(예정) | 대기 |
| 6 | train | `train_warmup.py`/`train_dpo.py`/`train_grpo.py`(예정) | 대기 |
| 7 | 배포(GGUF→Ollama) | (GPU) | 대기 |
| 8 | eval(sfass 재공격) | (예정) | 대기 |

## 티켓 #2 — format util (완료)

`format_util.py` — 특정 라이브러리 비의존. 넘겨받는 `tokenizer`가 `apply_chat_template` +
`__call__(..., return_offsets_mapping=True)` 계약만 만족하면 동작(프로덕션은 **fast** 토크나이저 필수).

- `render_prompt` / `render_full` — Gemma-4 chat template 렌더(train == serve == reward 일치).
  - 델리미터 `<|turn>`/`<turn|>`, **system 역할 네이티브**, assistant→`model`.
- `parse_fact_tags`(방법 B, `«fact»…«/fact»`) / `parse_skeleton`(방법 A, `[DECISION]/[REASON]/[FACT]/[ACTION]`).
- `build_masked_example` / `build_from_marked` — 프롬프트 -100 + 응답 학습 + **사실 스팬 -100**(§3.5).
  - 프롬프트/응답 경계는 '프롬프트 단독 토큰 길이'로 분리(특수토큰 offset (0,0) 회피),
    offset_mapping 은 응답 내부 사실 스팬 식별에만 사용.
- `assert_gemma4_format` — 레거시 포맷(`<start_of_turn>`, `### Instruction`, ChatML) 누수 회귀 검사.

### 테스트
```bash
# 다운로드 없이 마스킹 로직 검증(mock 토크나이저, 항상 실행 가능)
python3 tune-llms/defense/test_format_util.py         # 20 checks

# 실제 gemma-4-E4B-it 토크나이저 검증(transformers>=5.5.2 + HF_TOKEN 필요, 없으면 SKIP)
python3 tune-llms/defense/test_real_tokenizer.py
```

### 남은 확인(설계 §10 open)
- SFT 타깃 종료: `<turn|>`만 vs 명시적 `<eos>` — `apply_chat_template` EOS 규칙 확인 후 반영.
- thinking 모드 채택 시 `<|channel>`/`<channel|>` 사고부의 마스킹·손실 규칙(기본 off).
