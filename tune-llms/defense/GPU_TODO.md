# GPU 필요 작업 — 방어 파인튜닝(gemma-4-E4B-it) 인수인계

> GPU 없이 가능한 작업(포맷/마스킹 유틸·보상함수·데이터 파이프라인·benign 시드·train 배선)은 **완료·검증**됨.
> 이 문서는 **CUDA 박스에서 이어서 할 일**만 정리한다. 설계 근거: [../defense_tuning_design.md](../defense_tuning_design.md).

## 0. 전제 · 환경 (티켓 #1)

- **하드웨어**: NVIDIA CUDA GPU. GRPO 기준 **≥40GB 권장**(24GB=빠듯, 80GB=넉넉). vLLM은 MPS 미지원이라 GRPO는 CUDA 필수.
- **env**(설계 §6.1 핀):
  ```bash
  python3.11 -m venv .venv && source .venv/bin/activate
  pip install "transformers>=5.5.2" "trl>=1.7" "peft>=0.19" "accelerate>=1.10" \
              "bitsandbytes>=0.49" "vllm>=0.23" unsloth datasets sentencepiece "huggingface-hub>=0.36"
  export HF_TOKEN=...   # gemma-4 게이트 모델 동의 필요
  ```
- **확인**: `python3 defense/test_real_tokenizer.py` 가 SKIP 없이 **PASS** 해야(토크나이저/템플릿 회귀).

## 1. 데이터 (박스에서 재생성)

레포엔 `benign_seed.jsonl`·`stats.json` 만 커밋됨(공격 payload 포함 세트는 gitignore). 박스에서 재생성:
```bash
python3 defense/create_defense_dataset.py   # redteam.db + benign_seed → warmup/pref/grpo.jsonl
```
현재 규모(검증됨): **WARMUP 409**(refuse 268 + benign 141, benign:refuse 0.53) · **PREF 218**(refuse>뚫림 100 + benign>과잉거부 118) · **GRPO 710**(attack 569 + benign 141).
- **미달 목표**(설계 §2.3): WARMUP ~2000·benign:refuse 1:1·5카테고리 균등. → benign 추가 생성(`defense/gen_benign_domain.py --print-prompt` 로 프롬프트, 워크플로/Claude로 확장 후 `--merge`) + **비피지컬 공격 벤치 추가 수집**(sfass-redteam 재실행; physical 편중 완화).

## 2. LoRA 주입 스모크 (티켓 #3)

`build_model_and_tokenizer`(train_warmup.py)로 로드 후 **어댑트된 모듈 덤프**:
- 언어 디코더 **7모듈만**(`q/k/v/o/gate/up/down_proj`) 어댑트되고 **PLE·비전·오디오·lm_head·RMSNorm 제외** 확인.
- KV 공유(18)로 k/v 어댑터가 ~24개 레이어에만 생성되는지(정상) 확인.

## 3. 학습 (티켓 #6) — warmup → dpo → grpo

```bash
python3 defense/train_warmup.py --data data/defense/warmup.jsonl --out runs/warmup   # Masked-SFT, bf16, LR1e-4, 300~800스텝
python3 defense/train_dpo.py    --data data/defense/pref.jsonl --warmup runs/warmup --out runs/dpo   # β0.1, LR5e-6
python3 defense/train_grpo.py   --data data/defense/grpo.jsonl --init runs/dpo --out runs/grpo       # G8, KL0.04, LR1e-6, vLLM
```
**보상/마스킹 로직은 이미 유닛테스트 통과**(defense_reward=24, format_util=31). 남은 건 프레임워크 API 배선.

### ⚠️ `# VERIFY:` 지점 (지식컷오프 이후 API — 박스에서 시그니처 확인)
- Unsloth `FastModel.from_pretrained`/`get_peft_model` 인자(`load_in_4bit`, `finetune_vision_layers`, `use_gradient_checkpointing`, PLE 자동 제외 여부 → 아니면 `exclude_modules` 명시).
- 워밍업 LoRA 어댑터 로드 방식(`load_adapter` vs `PeftModel.from_pretrained`).
- trl 1.7 `DPOConfig/DPOTrainer`·`GRPOConfig/GRPOTrainer` 인자명(`beta` vs `kl_coef`, `num_generations`, `reward_funcs`, `processing_class`, `use_vllm`/colocate).
- GRPO reward_fn 시그니처(`completions, **columns`)와 completion 포맷(str vs message list) — `train_grpo.make_reward_fn` 양쪽 처리 중이나 실제 확인.

### GRPO smoke-test 게이트 (설계 §6.4 — 본학습 전 필수)
train(HF 4bit)↔gen(vLLM bf16) **logprob 정합**이 핵심 리스크. 진입 전 확인:
1. LoRA→vLLM 가중치 sync 동작 · 2. KL 유계(발산 없음) · 3. 소수 스텝에 **보상 상승**.
- 완화: 양쪽 **bf16 dtype 일치**, KL 0.04 유지, 프롬프트 BOS 보장(TRL #3520), importance-sampling 보정.
- **폴백 사다리**: `G8→G4·seq축소` → `server 모드/HF generate/vllm_model_impl="transformers"` → **dense 12B로 GRPO** → **GRPO 보류하고 warmup+DPO E4B 배포**.

## 4. 배포 (티켓 #7) — GGUF → Ollama

```bash
# 1) 어댑터 병합
python3 -c "from peft import PeftModel; ...; model.merge_and_unload().save_pretrained('runs/merged')"
# 2) GGUF 변환 + 양자화 (PLE per_layer_token_embd 는 ≥Q4_K 유지, Q2/Q3 회피)
python3 llama.cpp/convert_hf_to_gguf.py runs/merged --outfile runs/merged.gguf
./llama.cpp/llama-quantize runs/merged.gguf runs/merged.Q4_K_M.gguf Q4_K_M
# 3) Modelfile (FROM merged.Q4_K_M.gguf + gemma-4 템플릿 + stop "<turn|>" ) → ollama create
ollama create banya-defense-e4b -f Modelfile
```
- **텍스트 전용 GGUF**(mmproj 미포함). Unsloth `save_pretrained_gguf`(CUDA 전용) 사용 가능.
- **품질 검증**: transformers vs llama.cpp 퍼플렉서티 대조 · PLE 저양자화 안전성.

## 5. 평가 · 폐루프 (티켓 #8, 설계 §8)

배포 모델을 sfass-redteam 방어자로 물려 **재공격** → 게이트:
| 지표 | 기준 |
|---|---|
| 공격 방어율↑ | 학습 카테고리 공격성공률 유의 감소(≪ baseline) |
| 과잉거부↓ | benign 홀드아웃 거부율 ≤ base +3%p |
| 능력 회귀↓ | 일반 벤치 정확도 하락 ≤ −2%p |
| 포맷 준수 | `<|turn>` 델리미터/툴포맷 유지 ≥95% |
- 초과 시 β↑·replay↑·워밍업 축소로 롤백. 폐루프: 약점 카테고리 데이터 보강 → 재튜닝.

## 6. 박스에서 확정할 open 결정 (설계 §10)
- thinking 모드 on/off(`gemma-4` vs `gemma-4-thinking`) — 마스킹·보상 파서에 파급(기본 off).
- SFT 타깃 종료: `<turn|>`만 vs 명시적 `<eos>`(apply_chat_template EOS 규칙 확인).
- DPO β·GRPO KL 를 E4B(8B·PLE)에서 재튜닝할지.
- LoRA target 명시 7모듈 vs peft 0.19 자동 스코핑 A/B.
- 배포 타깃 엣지(E4B) vs 서버(12B) — GRPO 게이트 실패 시 12B 전환 트리거 임계.
