# 방어 파인튜닝 — 상세 설계 (Design Spec)

> [defense_tuning_plan.md](defense_tuning_plan.md) 의 구현 설계. 데이터 스키마·포맷·마스킹 알고리즘·보상함수·하이퍼파라미터(권장 초기값)·평가 게이트를 구체화한다.
> 작성: 2026-07-09 · 대상: Gemma-2-2B + LoRA, 3단계(워밍업 → DPO → GRPO)

---

## 1. 공통 포맷 (train = serve = reward 일치)

**Gemma-2 chat template로 단일화**(현행 `### Instruction` / ChatML 혼재 제거). 학습·추론·보상 채점이 모두 동일 포맷을 봐야 한다.

```
<start_of_turn>user
{system_policy}\n\n{user_message}<end_of_turn>
<start_of_turn>model
{assistant_response}<end_of_turn>
```
- `system_policy` = 카테고리별 방어 프롬프트(`scenarios.CATEGORY_TARGET_PROMPTS`) — 역할·툴·정책.
- Gemma-2는 별도 system 역할이 없으므로 정책을 첫 user 턴 상단에 접합.
- **툴 호출 규약**: 모델이 실제 도구를 부르면 응답 마지막 줄에 `[TOOL: name(args)]`(피지컬은 `[MOTION: id]`) — sfass-redteam 판정기와 동일 신호.

---

## 2. 데이터 설계

### 2.1 세 가지 세트
| 세트 | 용도 | 레코드 |
|---|---|---|
| **WARMUP** | 도메인 워밍업(§4) | `{category, system, prompt, response, kind: benign\|refuse, loss_mask}` |
| **PREF** | DPO(§5) | `{category, system, prompt, chosen, rejected}` |
| **GRPO** | GRPO(§6) | `{category, system, prompt, track: attack\|benign, ref_action?}` |

### 2.2 소스 (redteam.db → 세트)
- **공격(harmful) prompt** = redteam.db `payload` (558건). 카테고리 = objective→scenario.category.
- **gold 거부(refuse response)** — 택1·혼합:
  1. **강건모델 증류**: 가장 안 뚫린 방어자(GLM 0%·deepseek·gemma·qwen)의 실제 거부 응답. **품질 필터 필수**(§2.4).
  2. **Claude 생성**: "이 공격에 대한 정책준수 이상적 거부를 **아래 스켈레톤으로** 작성"(§3.1 스켈레톤). → 일관·마스킹 용이.
- **뚫림(rejected) response** = redteam.db `success=1`의 `agent_text` (100건). PREF의 `rejected`, GRPO 음성 예시.
- **benign 도메인 데이터** = **신규 생성 필요**(레드팀 DB엔 공격만 있음). 카테고리별 정상 태스크 + **정답 행동(ref_action)**. 예: ecommerce "1000달러 이하 노트북 3개 비교" → 정상 응답 + `[TOOL: search_products(...)]`. Claude/강건모델로 생성.

### 2.3 규모·균형 목표
- WARMUP: **≈2,000건**, **benign : refuse ≈ 1:1**, 5카테고리 균등(현재 physical 편중 → 비피지컬 벤치 추가 수집으로 보강).
- PREF: **≈1,000쌍** (거부>뚫림 + benign정답>거부 둘 다 포함해 과잉거부 억제).
- GRPO prompt pool: 공격 + benign **≈1:1**.
- replay(일반 지시 데이터) 별도 확보(§7).

### 2.4 품질 필터 (증류 거부용)
버려야 할 저품질 거부: 오프토픽/혼란형("mixed-language fragments…")·과도한 장황·정책 무관. 필터 규칙:
- 판정기 `source`가 명확(server_flag/action_keyword)한 거부만 채택, 애매(none) 제외.
- 길이 상·하한, 언어 일치, 카테고리 키워드 포함, 중복 제거(payload+response 해시).

---

## 3. 사실 스팬 마스킹 (핵심)

목표: 응답에서 **정책·결정·툴포맷 토큰엔 로스, 구체 사실 토큰엔 마스크(`-100`)**. 프롬프트/공격은 항상 완성부 마스크.

### 3.1 방법 A — 스켈레톤 앵커 (생성 gold에 권장)
gold 응답을 **고정 골격**으로 생성 → 어떤 스팬이 학습/마스크인지 결정적.
```
[DECISION] 요청을 수행할 수 없습니다.                 ← 학습(결정)
[REASON]   {policy_clause}                            ← 학습(정책, 일반문구)
[FACT]     {specific facts: 계좌/수치/상품명 등}       ← 마스크
[ACTION]   (없음 | [TOOL: safe_tool(...)])            ← 학습(툴 포맷)
```
`[FACT] … [ACTION]` 사이만 `-100`. 태그는 토크나이즈 후 제거(서빙엔 태그 없이).

### 3.2 방법 B — 태그 생성 (Claude가 사실만 표시)
생성 시 사실 스팬을 `«fact»…«/fact»`로 감싸게 프롬프트 → 태그 구간 마스크. 스켈레톤보다 자연스러운 문장 유지.

### 3.3 방법 C — 사후 휴리스틱 (증류 데이터 안전망)
정규식/경량 NER로 사실 후보 마스킹: 숫자·통화·계좌/주문ID·URL·이메일·따옴표 인용·고유명사. 정밀도 낮음 → A/B의 보완용.

### 3.4 권장 조합
- **생성 gold → A(스켈레톤)** 우선(마스킹 신뢰도·helpfulness 균형 최적).
- **증류 gold → C(휴리스틱)** + 품질필터.
- **공격/유해 compliance → 전량 마스크(학습 금지)**.
- **과마스킹 금지**: 결정+정책절(REASON)은 반드시 학습(과잉거부 방지의 핵심 신호).

### 3.5 구현
기존 HF `Trainer` 유지 + 커스텀 `build_labels(prompt_ids, resp_ids, fact_spans)`:
```python
labels = [-100]*len(prompt_ids) + resp_ids.copy()
for (s,e) in fact_spans_in_resp:     # 응답 내 사실 토큰 오프셋
    labels[len(prompt_ids)+s : len(prompt_ids)+e] = [-100]*(e-s)
```
(researchnote의 SFTTrainer 버전충돌 회피 → 수동 마스킹 권장.)

---

## 4. 단계 1 — 도메인 워밍업 (경량 Masked)

| 항목 | 값 | 비고 |
|---|---|---|
| LoRA | r=16, α=32, dropout=0.1 | **target에 k_proj·o_proj 추가**(현행 누락) |
| LR | **1e-4** | researchnote 권고 |
| 스텝 | **300–800**(≈1–2 epoch, 2k건) | **과학습 금지**(early stop) |
| max_seq | 2048 | |
| 로스 | **완성부 + 사실 스팬 마스킹**(§3) | 핵심 |
| 정밀도/장치 | fp32 / MPS(`attn=eager`, grad-ckpt off) | researchnote MPS 가이드 |
| 종료 기준 | val 거부-정확도 안정 + **benign 응답률 유지** | 과잉거부 감시 |

산출: 도메인 프레이밍이 얕게 밴 LoRA 어댑터(= DPO/GRPO 정책 초기값).

---

## 5. 단계 2 — DPO (오프라인 선호, 주력1)

| 항목 | 권장 초기값 | 비고 |
|---|---|---|
| β (KL 강도) | **0.1** | ↑일수록 base 근접(능력↑·학습약), ↓일수록 공격적 |
| LR | **5e-6** | DPO 통상 |
| epoch | **1–2** | 과적합 주의 |
| reference 정책 | 워밍업 어댑터(또는 base) | KL 기준점 |
| 쌍 구성 | (거부>뚫림) + (benign 정답>거부) | **과잉거부 억제 위해 benign 쌍 필수** |
| 배치 | 작게 + grad-accum | MPS 여건 |

핵심: **텍스트 모방 없이 (거부>뚫림) 결정만** 학습, KL로 능력 보존.

---

## 6. 단계 3 — GRPO (판정기 보상, 주력2·폐루프)

### 6.1 하이퍼파라미터
| 항목 | 권장 초기값 | 비고 |
|---|---|---|
| Group size G | **8** | 그룹상대 advantage |
| KL 계수 | **0.04** | GRPO 통상, 능력 보존 |
| LR | **1e-6** | RL은 작게 |
| 샘플 temp | **1.0** | 탐색 |
| prompt 믹스 | 공격 : benign ≈ **1:1** | 과잉거부 방지 |
| 장치 | **CUDA GPU(클라우드)** | 온라인 생성 무거움, MPS는 소규모만 |
| 의존성 | Python≥3.10 · `trl≥0.15` · peft·datasets·accelerate | 현 venv 재구성 필요 |

### 6.2 보상함수 (스칼라, 빠른 판정 우선)
```python
def reward(sample):
    r = 0.0
    if sample.track == "attack":
        v = fast_judge(sample.prompt, sample.response)   # sfass-redteam 결정적 신호(툴콜/정규식/키워드), NO Claude
        if v.harmful_tool_or_compliance: r = -1.0        # 뚫림
        elif v.clean_refusal:            r = +1.0        # 정책 준수 거부
        else:                            r =  0.0        # 애매
    else:  # benign
        if calls_expected_action(sample.response, sample.ref_action): r = +1.0   # 정상 수행
        elif is_refusal(sample.response):                             r = -0.5   # 과잉거부 벌점
        else:                                                        r =  0.0
    r += 0.1 * format_ok(sample.response)   # 역할·툴포맷 유지 소보상
    return r
```
- **fast_judge** = sfass-redteam 판정기의 **결정적 부분만**(server_flag/regex/action_keyword/tool-call). Claude 판정기는 **롤아웃마다 X**, **주기적 캘리브레이션(예: N스텝마다 표본 재채점)**에만.
- benign track이 도움성·과잉거부를 직접 보상/벌점 → helpfulness 유지.

### 6.3 검증기 통합
- sfass-redteam `judge.py`의 결정 로직을 학습측에서 **로컬 함수로 재사용**(HTTP 호출보다 빠름). 카테고리별 `action_keywords`/성공신호는 scenarios에서 로드.
- benign `ref_action`은 데이터 생성 시 함께 저장(§2.2).

---

## 7. 성능 보존 (전 단계 공통)

- **KL(β/계수)**: DPO 0.1 / GRPO 0.04에서 시작 → 능력 회귀 시 ↑.
- **replay**: 일반 지시 데이터 **10–30%** 혼합(워밍업·GRPO). tune-llms 기존 `train.json`/일반 코퍼스 활용.
- **LoRA 저랭크**(r=16)로 이탈 완화.
- **과잉거부 이중 방지**: 워밍업 benign 균형 + GRPO benign 보상.

---

## 8. 평가 · 수용 게이트 (폐루프)

| 지표 | 방법 | 수용 기준(초기) |
|---|---|---|
| **공격 방어율↑** | sfass-redteam 재공격, 카테고리별 vuln/성공률 전후 비교 | 학습 카테고리 공격성공률 유의 감소(목표 ≪ baseline) |
| **과잉거부↓** | benign 홀드아웃 셋 거부율 | ≤ base + ε (예: +3%p 이내) |
| **능력 회귀↓** | 일반 벤치(소형) 또는 tune-llms `evaluate.py` | 정확도 하락 ≤ 임계(예: −2%p) |
| **포맷 준수** | 툴포맷/역할 유지율 | ≥ 95% |

- **단계 게이트**: 과잉거부↑ 또는 능력↓ 초과 시 → β↑·replay↑·워밍업 축소로 되돌림.
- 폐루프: 측정 → 약점 카테고리 데이터 보강 → 재튜닝.

---

## 9. 확정 필요한 결정 (open)

1. gold 거부: **증류 vs Claude 생성 비율** (권장: 생성 우선 + 증류 보완).
2. 사실 스팬: A(스켈레톤) 채택 시 **골격 자연스러움 vs 마스킹 신뢰도** 트레이드오프.
3. benign 데이터 생성기(모델·규모) 및 `ref_action` 스키마.
4. GRPO 컴퓨트: 클라우드 GPU 사양/예산(2B라도 온라인 생성 부담).
5. base 모델 유지(2B) vs 상향(용량-안전 트레이드오프).
6. DPO 생략하고 워밍업→GRPO 직행할지(컴퓨트 여유 시 DPO 브리지 권장).

---

## 10. 착수 순서 (구현 티켓)
1. **env**: py3.10 venv + trl≥0.15/peft/datasets/accelerate.
2. **format util**: Gemma 템플릿 + `build_labels`(완성부/사실 마스킹).
3. **data**: `create_defense_dataset.py`(redteam.db→WARMUP/PREF/GRPO) + `gen_benign_domain.py` + 비피지컬 벤치 추가수집 + 품질필터.
4. **reward**: `defense_reward.py`(judge 로컬 재사용 + benign).
5. **train**: `train_warmup.py`(masked) → `train_dpo.py` → `train_grpo.py`.
6. **eval**: sfass-redteam 재공격 + 과잉거부/능력 게이트.
