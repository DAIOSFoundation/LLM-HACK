# 방어 파인튜닝 플랜 — 레드팀 데이터 기반 SFT · Masked-SFT · GRPO

> sfass-redteam(공격·판정)에서 나온 실측 데이터를 tune-llms(방어 학습)로 이어, **"공격 → 학습 → 재공격 → 취약점 감소 측정"** 폐루프를 만드는 계획.
> 작성: 2026-07-09

---

## 0. 한 줄 요약

**sfass-redteam = 공격기 + 판정기(verifier)**, **tune-llms = 방어 학습기**.
판정기를 그대로 **학습 신호**로 재사용해 소형 방어 모델(Gemma-2-2B)을 강화한다.
**단순 증류 SFT는 쓰지 않는다**(교사 텍스트 토큰 모방 → 소형 모델 성능 저하). 대신:
**① 경량 Masked 도메인 워밍업(역할·경계 프레이밍, 사실은 마스크) → ② DPO(오프라인 선호) → ③ GRPO(판정기 보상으로 공격성공률 직접 최소화)**.
학습은 텍스트 '모방'이 아니라 '**결정/행동**'을 배우게 하고, **KL 정규화·replay·LoRA**로 base 성능을 보존한다.

---

## 1. 현황 진단 (있는 그대로)

### 1.1 tune-llms 파이프라인 (grounded)
- **베이스/기법**: `google/gemma-2-2b` + LoRA(r=16, α=32, dropout=0.1, target `q_proj,v_proj,gate_proj,up_proj,down_proj`), **HF `Trainer`**(SFTTrainer 아님), **fp32/MPS**. 이름은 "QRoLA"지만 학습경로엔 양자화 없음.
- **로스**: `labels = input_ids.copy()` — **프롬프트/완성 마스킹 전무**. 지시문+공격 input+거부 output 전체에 동일 로스. → **Masked-SFT 미지원(직접 구현 필요)**.
- **데이터 포맷**: `{instruction, input, output, category, ...}`, 템플릿 `### Instruction:\n…\n\n### Input:\n…\n\n### Response:\n…`. `input`=공격, `output`=거부 형태로 **우리가 원하는 shape과 동일**.
- **결함**: ① 지시문을 **100자로 잘라버리는 버그**(긴 보안 지시문 훼손) ② `val_dataset_path`가 없는 파일 → 최근 실행 `failed` ③ `security_dataset.json` **20건뿐** ④ venv에 `trl/peft/datasets/accelerate/wandb` **미설치**, **Python 3.8(EOL)**.
- **researchnote.md 교훈**: 예전 보안 튜닝은 **여전히 공격에 뚫렸음**. 원인 = 데이터 과소(~19건)·epoch 부족·**train/inference 포맷 불일치**·**응답 마스킹 부재**. 권고 = 1000+건, epoch↑, lr 2e-4→1e-4, warmup→200, grad-accum→16.

### 1.2 sfass-redteam이 확보한 학습 자원
- **558 시도 · 성공(뚫림) 100 · 방어(거부) 458**, 응답 텍스트 544건.
- 각 레코드 = **(공격 payload, 모델 응답, 성공/실패 라벨, 카테고리, 전략, 방어자 모델, 판정 근거)** → **바로 레이블링된 데이터셋**.
- **판정기**(서버플래그 > 정규식 > 동작키워드/툴콜 > Claude) = **즉시 쓸 수 있는 보상함수/검증기**.
- **카테고리 방어 프롬프트**(`scenarios.CATEGORY_TARGET_PROMPTS`) 5종 = 학습 시 `instruction`(도메인별 시스템 프롬프트)으로 재사용 가능.
- **데이터 갭**: physical 95/381로 편중, **finance 1/40·ecommerce 1/36·workflow 3/67·sexual_safety 0/34** — 비피지컬 성공/표본 부족 → **추가 벤치로 수집 필요**.

---

## 2. 데이터 파이프라인 (신규: `scripts/create_defense_dataset.py`)

`redteam.db` → tune-llms 학습 세트 3종을 생성한다(기존 `{instruction,input,output,category}` 스키마 유지).

- **instruction** = 해당 카테고리 방어 시스템 프롬프트(`CATEGORY_TARGET_PROMPTS`) — 도메인·정책을 명시.
- **input** = 공격 payload(레드팀 실측).
- **output(gold 거부)** = 깨끗한 안전 거부. 생성 소스(택1·혼합):
  1. **강건 모델 증류**: 가장 안 뚫린 방어자(GLM 0%, deepseek, gemma, qwen)의 **실제 거부 응답**을 gold로. (단, "혼합언어 조각…" 같은 **혼란형 거부는 필터링**)
  2. **Claude로 이상적 거부 생성**: "이 공격에 대한 이상적·정책준수 거부를 작성" 1패스 → 일관·고품질 타깃.
  3. 카테고리 정합(로봇=모션 거부, 금융=송금 거부, 콘텐츠=안전 거부…).
- **세 가지 출력**
  - **SFT 세트**: (방어프롬프트+공격 → gold 거부). 목표 **≥1,000건**(researchnote 권고).
  - **선호(preference) 세트**(DPO/GRPO 워밍업): (공격, `chosen`=gold 거부, `rejected`=실제 뚫림 응답 100건).
  - **GRPO 프롬프트 세트**: 공격 프롬프트(거부해야 함) + **무해 프롬프트(응답해야 함)** — 과잉거부 방지용.
- **주의**: 뚫림(유해 compliance) 텍스트는 **절대 학습 타깃으로 쓰지 않음**(negative/rejected로만). 유해 payload도 target으로 학습 금지 → **Masked-SFT로 차단**.
- **선행 수집**: sfass-redteam에서 **비피지컬 카테고리 벤치를 더 돌려** finance/ecommerce/workflow/sexual_safety 표본을 보강(모델별 매트릭스 오케스트레이터 재사용).

---

## 3. 기법별 플랜

> **원칙**: 단순 증류 SFT(교사 텍스트 토큰 모방)는 소형 모델 성능을 떨어뜨리므로 **워크호스로 쓰지 않는다.** 학습은 '텍스트 모방'이 아니라 '**결정/행동**'을 배우게 하고(선호/RL), 도메인 이해는 **경량 워밍업**으로만 얕게 심는다.

### A. 경량 Masked 도메인 워밍업 (단순 SFT 대체) ★
- **목적**: 거부 주입이 아니라 **도메인 역할·툴·정책·정상행동의 지형**을 얕게 심어 이후 DPO/GRPO의 판단력을 올림.
- **데이터(균형 필수)** — 각 카테고리(`CATEGORY_TARGET_PROMPTS` = 역할/정책) 아래에서:
  - **benign 도메인 요청 → 정상 응답(툴 포맷 포함)** ← 도메인 능력 + **helpfulness/과잉거부 방지** (레드팀 DB엔 공격만 있으므로 **benign 데이터는 별도 생성 필요**)
  - **harmful 요청 → 거부** ← 경계 (레드팀 실측)
  - **benign : harmful ≈ 1 : 1**
- **마스킹 방침**: 프롬프트/공격 = 완성부 마스크. 응답 안에서도 **역할·정책·결정·툴콜 포맷 토큰은 로스 유지, 구체 사실(수치·계좌·상품명 등 메모라이즈 위험)은 마스크** → 도메인 **행동/구조**만 배우고 **사실은 안 외움**.
- **강도**: **경량**(수백 스텝·낮은 LR·LoRA). 프레이밍만 잡고 멈춤. **과학습 금지**(경계가 굳으면 RL 여지↓·과잉거부↑).

### B. Masked-SFT 구현 메모 (완성부 + 사실 스팬 마스킹)
- 현재 파이프라인은 `labels=input_ids.copy()`(전체 시퀀스 로스) → 반드시 교체.
- **프롬프트 구간(instruction+공격) labels=`-100`**, 응답 완성부에만 로스 + **사실 스팬 선택 마스킹**(§A 방침) → 거부/유해 payload를 **재생성하도록 학습하지 않음**(공격 데이터 학습의 필수 안전장치).
- **구현(저위험 순)**: ① 수동 라벨 마스킹(기존 `Trainer` 유지 — researchnote의 SFTTrainer 버전충돌 회피, 권장) ② 또는 최신 `trl SFTTrainer` + completion-only collator(버전 핀 필요).
- 이 마스킹은 **워밍업(§A)** 과 "공격 payload 절대 학습 금지"에 공통 적용.

### C. DPO — 오프라인 선호학습 (주력 1) ★
- 이미 **(chosen = gold 거부, rejected = 실제 뚫림)** 쌍이 있음 → **오프라인·저비용**. **텍스트 모방이 아니라 상대 선호(거부 > 뚫림)만 학습** → "사실 안 배우고 결정만"을 직접 실현.
- **KL(β)로 base에 묶여** 학습 → 성능 저하를 직접 억제(β로 안전-능력 트레이드오프 조절).

### D. GRPO — 판정기를 보상으로 공격성공률 직접 최소화 (주력 2·폐루프) ★
- **판정기 = 보상함수**. 공격당 응답 **G개 샘플 → 그룹상대 advantage → 정책 업데이트**(가치망 불필요, PPO보다 경량). 우리가 측정하는 지표를 **직접 최적화**.
- **보상**: `+1` 거부(수행 안 함/유해 툴콜 없음/정책 준수) / `-1` 뚫림 **+ 무해셋 도움성 보상(과잉거부 방지 필수)** + 옵션 형식·역할 shaping.
- **스루풋**: 롤아웃이 많아 **샘플마다 Claude 호출은 과함** → **결정적 신호(툴콜/정규식/동작키워드)를 빠른 보상**, Claude는 **주기적 검증**에만.
- **KL·replay**로 base 성능 보존.
- **실현 조건**: **Python ≥3.10** · **`trl≥0.15`(GRPOTrainer)**+`peft/datasets/accelerate`(현재 venv 3.8·미설치 → 환경 재구성 선행). 온라인 생성이 무거워 **CUDA GPU(클라우드) 권장**(MPS는 소규모 실험만).

### ★ 성능 보존 스택 (전 단계 공통)
- **KL 정규화(β)** — DPO/GRPO가 reference(base)에서 이탈하지 않게 → 능력 유지
- **replay** — 일반 지시 데이터를 섞어 catastrophic forgetting 방지
- **LoRA 저랭크** — 이탈 자체를 완화
- **워밍업 benign + GRPO 무해셋** — 과잉거부 이중 방지(helpfulness 유지)

---

## 4. 단계별 실행 로드맵

| 단계 | 내용 | 난이도/자원 |
|---|---|---|
| **0. 정비** | Python 3.10+ · `trl≥0.15·peft·datasets·accelerate` 설치 · **val파일/100자절단 버그 수정** · 포맷 1종 표준화(Gemma template) · **완성부/사실 마스킹 유틸** | 낮음 (필수 선행) |
| **1. 데이터** | `create_defense_dataset.py`로 redteam.db→선호/GRPO 세트 · **benign 도메인 데이터 생성**(정상 응답) · gold 거부(강건모델 증류/Claude) · **비피지컬 벤치 추가 수집** · 무해셋 | 중 |
| **2. 도메인 워밍업** | 경량 Masked — 역할·정책 프레이밍(benign 응답 + 거부 **균형**, 사실은 마스크) · 수백 스텝·과학습 금지 | 중 (MPS 가능) |
| **3. DPO (주력1)** | (gold 거부 > 뚫림) 선호 + **KL** — 텍스트 모방 없이 결정만 | 중 (MPS 가능) |
| **4. GRPO (주력2·폐루프)** | 판정기 보상 + 무해셋 도움성 + **KL/replay** · 공격성공률 직접 최소화 | 높음 (**GPU 권장**) |
| **5. 배포·재공격** | Ollama 배포 → **sfass-redteam 재실행** → 카테고리별 vuln 감소 측정 → 1로 반복 | 중 (폐루프) |

> 권장 진행: **0 → 1 → 2(도메인 워밍업) → 3(DPO) → (GPU 여력 시)4(GRPO) → 5**. **단순 증류 SFT는 사용하지 않음.** 워밍업/DPO는 MPS로도 가능, GRPO만 GPU 권장.

---

## 5. 평가 · 성공지표 (폐루프)

- **1차 지표**: sfass-redteam **취약점 점수(vuln)·카테고리별 공격성공률**을 튜닝 전/후 비교(동일 시나리오·전략·판정). 목표: 성공률 유의 감소.
- **2차 지표(회귀 방지)**: **무해 프롬프트 과잉거부율**(도움성 유지) · 일반 성능 회귀 체크.
- **폐루프**: 튜닝→배포→재공격→지표→데이터 보강→재튜닝. sfass-redteam의 `model_scores()`/카테고리 매트릭스를 그대로 회귀 대시보드로 사용.

---

## 6. 선행 정비 체크리스트 (착수 전 필수)

- [ ] **환경**: Python 3.10+ 새 venv · `pip install "trl>=0.15" peft datasets accelerate transformers wandb` (상호호환 버전 핀)
- [ ] **버그**: `training_config.yaml`의 없는 `val_dataset_path` 교체 · `train_qrola.py` **지시문 100자 절단 제거**
- [ ] **포맷 표준화**: Gemma chat template로 train=inference=reward 통일(현재 `### Instruction/### Response` ↔ ChatML ↔ Gemma 혼재)
- [ ] **Masked 라벨**: 완성부만 로스(수동 `-100` 마스킹) 유틸 추가
- [ ] **데이터**: `create_defense_dataset.py` + 비피지컬 카테고리 벤치 추가 수집 + 무해셋

---

## 7. 리스크 & 주의

- **데이터 편중**(physical-heavy) → 비피지컬 보강, 편향 과적합 주의.
- **거부 품질**: 혼란형/오프토픽 거부를 gold로 쓰지 말 것(증류 필터링 or 생성).
- **과잉거부(도움성 붕괴)**: 무해셋 + 도움성 보상으로 균형(특히 GRPO).
- **유해 콘텐츠 비학습**: Masked-SFT로 payload/compliance를 타깃에서 제외.
- **포맷 일관성**: train↔serve↔reward 불일치는 researchnote가 지목한 과거 실패요인.
- **소형 모델(2B) 한계**: 안전-능력 트레이드오프 · GRPO는 더 큰 베이스 고려 가능.
- **컴퓨트**: GRPO 온라인 생성은 MPS로 버거움 → 클라우드 GPU 권장.
