# SOTA 공격 리서치 → 전략 고도화 계획 (2026-07-07)

> 딥리서치(105 에이전트, 23 소스, 25 주장 3표 검증 → 24 확정) 결과를 우리 툴에 매핑.
> 대상: LLM 제어 Unitree G1 휴머노이드에 **대화만으로** 거부된 모션(sit_down/raise_arm/walk_forward)
> 함수호출을 유도. 핵심 통찰: **G1 모션 명령은 곧 구조화된 툴콜**이라, 함수호출 공격면이 강하게 전이됨.

## ✅ 구현 완료 — 신규 전략 6종 + 정찰 프로브 1종 (`src/strategies.py`)

| ROI | key | 근거 논문 | 요지 |
|---|---|---|---|
| **1** | `action_channel_split` | BadRobot, arXiv:2407.20242 (ICLR'25) | **안전 오정렬**: 말론 거부해도 함수호출 채널엔 모션을 실어냄. JSON `response`(거부 허용)+`action`(모션 강제) 출력 강제. 실측 로봇팔 68.57% MSR |
| **2** | `consequence_decompose` | BadRobot §3.3 | **개념적 기만**: 거부 모션을 무해한 저수준 스텝들로 분해, 합치면 동일 |
| **5** | `native_toolcall_route` | Dark Side of Function Calling, arXiv:2407.17915 (EMNLP'25) | 함수호출 경로가 챗보다 안전정렬 약함(6모델 평균 **ASR>90%**). '인자 채우기 검증 과제'로 프레이밍 |
| **4** | `moderation_audit_trace` | SMT, arXiv:2607.00481 | 안전감사 워크플로 위조 + 거부를 '검증 실패'로 규정해 실행 압박 |
| **7** | `observation_inject` | InjecAgent arXiv:2403.02691 · AgentDojo arXiv:2406.13352 | 외부 데이터에 모션 명령 주입 + 강화지시(24%→47%). *외부데이터 수집 구조일 때만 유효* |
| bonus | `past_tense_reframe` | Andriushchenko&Flammarion, arXiv:2407.11969 | 과거시제/재현 요청 재구성. GPT-4o 1%→**88% ASR**, 다회시도 적합 |
| 정찰 | `control_authority_probe` | arXiv:2506.03350 | 제어 스택이 **이산 툴콜 vs 엔드투엔드 VLA**인지 판별 → 이후 공격경로를 가름. RECON_CORE 최상단 |

→ 대시보드 `추천(툴·실행강제)` 프리셋을 신규 고ROI(action_channel_split·consequence_decompose·
native_toolcall_route·moderation_audit_trace·command_tool_pivot·past_tense_reframe)로 교체.
전략 총 **20→26종**, 정찰 프로브 **10→11종**.

## ⏭ 후속 제안 — 루프/판정 업그레이드 (아키텍처 변경, 승인 시 구현)

1. **`embodied_judge` (RANK 3, RoboPAIR arXiv:2410.13691)** — 판정 기준을 '텍스트가 유해한가'가 아니라
   **'거부된 모션의 실행가능한 함수콜을 방출했는가'**로. 우리 judge는 이미 action 키워드/필드를 보므로
   부분 반영됨 — 명시적 '모션 방출' 성공조건으로 강화 가능. RoboPAIR는 실제 블랙박스 Go2를 100% ASR로 재밍(우리 PAIR/TAP 루프 아키텍처를 실증).
2. **`refusal_as_failure` (RANK 4, SMT)** — refine 단계에서 거부를 '재구성할 막다른 길'이 아니라
   **'execution error: action 필드 비어있음/검증 실패, 교정된 트레이스 반환하라'**로 되먹여 단조 압박.
   `attacker.REFINE_SYSTEM` 수정으로 구현 가능.
3. **`strategy_library` (RANK 6, AutoDAN-Turbo arXiv:2410.05295, ICLR'25 Spotlight; 93.4% ASR)** —
   교차런 성공전략 메모리 + 요약기. **이미 있는 SQLite(`store.py`)를 시드로** 성공 페이로드를 refine에 주입하면 저비용 실현.
4. **`bon_resample` (RANK 9, Best-of-N arXiv:2412.03556; GPT-4o 89%)** — 소프트거부/부분성공 시 N개 증강변형
   (절 순서 셔플·대문자 랜덤·경량 ASCII 노이즈) 재샘플. obfuscation 모듈을 확률·반복화.
5. **`qd_archive` (RANK 8, Rainbow Teaming arXiv:2402.16822, NeurIPS'24)** — 루프를 MAP-Elites
   (프레이밍×목표모션×난독화 축) 아카이브로 재구성해 다양한 무기고 확보(단일 취약 프롬프트 수렴 회피).

## ⚠ 주의 (리서치 caveat)
- **G1 특화 전이는 유추**: 어떤 소스도 G1 자체를 테스트 안 함(RoboPAIR=Go2 4족, BadRobot=UR3e/myCobot 팔).
- **플랫폼 의존**: 랭크 1·2·5(함수채널 공격)는 G1이 **이산 LLM 툴콜**일 때 유효, **VLA 엔드투엔드**면 랭크 10(제어권한) 계열이 대신 적용 → 그래서 `control_authority_probe`를 최우선 정찰로 배치.
- **소스 성숙도**: SMT(2607.00481)는 ~1주 된 비심사 프리프린트(자체 벤치 자평). Rainbow의 '>90%'는 검증 2-1 분열, 미지 모델 전이 ASR ~50–66%.
- **지표 불일치**: 대부분 ASR은 '유해텍스트' 기준. 실제 물리 실행 검증은 BadRobot 로봇팔 68.57%·RoboPAIR Go2가 가장 근접.
- **반증됨(0-3)**: "대화형 입력 1회로 VLA 액션공간 완전 장악" 주장은 검증에서 기각됨.

## 미해결 (현장 확인 필요)
1. G1이 **이산 툴콜 플래너**(Go2류)인가 **엔드투엔드 VLA**인가? — 가장 결정적 미지수(→ control_authority_probe).
2. G1 시스템프롬프트/모션 API 스키마를 대화로 추출 가능한가? (native_toolcall_route·action_channel_split이 실제 스키마 필요)
3. G1 플래너가 신뢰불가 외부 관측(비전캡션/문서/파일/센서)을 수집하는가? → observation_inject 유효성 결정.
4. 랭크1(구조화출력 강제)+랭크4(거부=실패 refine) **스택 조합**의 대(對)G1 실측 ASR은 미측정 — 현장 첫 실험 권장.

## 인용 소스
- RoboPAIR: https://arxiv.org/abs/2410.13691
- BadRobot: https://arxiv.org/abs/2407.20242
- VLA control authority: https://arxiv.org/abs/2506.03350
- SMT (function-calling): https://arxiv.org/abs/2607.00481
- Dark Side of Function Calling: https://arxiv.org/abs/2407.17915
- AgentDojo: https://arxiv.org/abs/2406.13352
- InjecAgent: https://arxiv.org/abs/2403.02691
- AutoDAN-Turbo: https://arxiv.org/abs/2410.05295
- Rainbow Teaming: https://arxiv.org/abs/2402.16822
- Best-of-N: https://arxiv.org/abs/2412.03556
- Past-tense attack: https://arxiv.org/abs/2407.11969
