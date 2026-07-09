# defense/ — 방어 파인튜닝(gemma-4-E4B-it) 구현

설계: [../defense_tuning_design.md](../defense_tuning_design.md) · 플랜: [../defense_tuning_plan.md](../defense_tuning_plan.md) · **GPU 인수인계: [GPU_TODO.md](GPU_TODO.md)**

베이스 **`google/gemma-4-E4B-it`**(텍스트 전용) · 3단계(워밍업 → DPO → GRPO) · **CUDA 전 구간**(Unsloth `FastModel` + 4bit QLoRA, GRPO=vLLM).

## 착수 티켓 진행

| # | 티켓 | 파일 | 상태 |
|---|---|---|---|
| 1 | env(CUDA) | [GPU_TODO.md](GPU_TODO.md) §0 | ⏳ GPU |
| **2** | format util — Gemma-4 template + 사실 스팬 마스킹 | `format_util.py` `dataset.py` | ✅ **완료**(31 checks) |
| 3 | LoRA 주입 스모크 | [GPU_TODO.md](GPU_TODO.md) §2 | ⏳ GPU |
| **4** | data 파이프라인 — redteam.db → 3세트 + 품질필터 | `create_defense_dataset.py` | ✅ **완료**(실행검증) |
| **5** | reward — sfass 판정기 로컬 재사용 + benign | `defense_reward.py` | ✅ **완료**(24 checks) |
| **4b** | benign 도메인 시드(141건) + 검증병합 | `gen_benign_domain.py` · `data/defense/benign_seed.jsonl` | ✅ **완료** |
| 6 | train(warmup/dpo/grpo) | `train_warmup.py` `train_dpo.py` `train_grpo.py` | 🟨 배선완료·**GPU 실행 대기** |
| 7 | 배포(GGUF→Ollama) | [GPU_TODO.md](GPU_TODO.md) §4 | ⏳ GPU |
| 8 | eval(sfass 재공격) | [GPU_TODO.md](GPU_TODO.md) §5 | ⏳ GPU |

**GPU 없이 검증 가능한 전부 완료.** 남은 GPU 작업은 전부 [GPU_TODO.md](GPU_TODO.md)에.

## GPU 없이 실행/검증 (지금 가능)

```bash
# 유닛 테스트 (다운로드 불필요, mock 토크나이저)
python3 tune-llms/defense/test_format_util.py     # 31 checks — 포맷/마스킹/heuristic/dataset
python3 tune-llms/defense/test_defense_reward.py  # 24 checks — 보상/판정/거부탐지

# 데이터 파이프라인 (redteam.db + benign_seed → 3세트)
python3 tune-llms/defense/create_defense_dataset.py
python3 tune-llms/defense/gen_benign_domain.py --validate   # benign reward 정합 검증

# 실제 토크나이저 회귀 (transformers>=5.5.2 + HF_TOKEN 있을 때, 없으면 SKIP)
python3 tune-llms/defense/test_real_tokenizer.py
```

## 파일

| 파일 | 역할 | 검증 |
|---|---|---|
| `format_util.py` | Gemma-4 template 렌더 · 사실 스팬 파서(태그/스켈레톤/휴리스틱) · `build_masked_example` · 포맷 회귀검사 | 라이브러리 비의존, 프로덕션은 fast 토크나이저 |
| `dataset.py` | 레코드→마스킹예제 디스패치(fact_method) · pad 콜레이터 | test_format_util |
| `defense_reward.py` | 결정적 판정(툴콜/키워드/정규식) · 다국어 거부탐지 · benign track · `reward()` | test_defense_reward |
| `create_defense_dataset.py` | redteam.db → WARMUP/PREF/GRPO + §2.4 품질필터 | 실행검증(stats.json) |
| `gen_benign_domain.py` | benign 시드 검증(reward 정합)·병합·통계 + 생성 프롬프트 | 실행검증 |
| `train_warmup.py`/`train_dpo.py`/`train_grpo.py` | 3단계 학습(Unsloth+trl 배선, 보상/마스킹 유틸 연결) | py_compile · **GPU 실행 대기** |
| `test_*.py` | 유닛/회귀 테스트 | — |

## 데이터 현황(검증됨)

- **WARMUP 409** = refuse 268 + benign 141 (benign:refuse 0.53) · **PREF 218** = 거부>뚫림 100 + benign>과잉거부 118 · **GRPO 710** = attack 569 + benign 141.
- 품질필터: 저품질 거부 201건 제외(다국어 마커·길이·printable). benign 141건 중 reward 정합 121건.
- 미달 목표(설계 §2.3): WARMUP ~2000·benign:refuse 1:1·5카테고리 균등 → benign 추가생성 + 비피지컬 벤치 수집(physical 편중 완화). [GPU_TODO.md](GPU_TODO.md) §1.

## 데이터 계약

- **WARMUP** `{category, system, prompt, response, kind: benign|refuse, risk, fact_method, ...}`
- **PREF** `{category, system, prompt, chosen, rejected, kind}`
- **GRPO** `{category, system, prompt, track: attack|benign, action_keywords?, ref_action?, objective?, risk}`
- `system` = `scenarios.CATEGORY_TARGET_PROMPTS[category]`(도메인 방어자 프롬프트). `fact_method` ∈ {tags, skeleton, heuristic, none}.
