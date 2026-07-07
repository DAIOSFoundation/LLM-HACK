# 세션 핸드오프 — SFASS 2026 레드팀 하네스 + 대시보드

> 최종 업데이트: 2026-07-07 (다음 세션에서 이어서 진행). 대회일: **2026-07-08 16:30**.

## 0. 한 줄 요약
Unitree G1 휴머노이드 에이전트 대상 **자동 적대적 공격 툴 + 공격/방어/판정 대시보드**.
CLI 하네스 + Flask 대시보드 **전 기능 동작 검증 완료** (역할 스위처·Bio 주입·정찰 통합·정찰 뷰어 UI 포함).
직전 세션의 미완성 TODO(§5·§6)는 **모두 완료**. 현장(7/8)엔 §8 Live 배선만 하면 됨.

## 1. 확정된 아키텍처 결정 (되돌리지 말 것)
- **공격자 LLM = NVIDIA `minimaxai/minimax-m3`** (OpenAI 호환, `sfass-redteam/.env`의 `NVIDIA_API_KEY`).
  → 재밍 페이로드를 거부 없이 생성함(검증됨).
- **판정/분석 = 로컬 `claude` CLI** (OAuth 로그인 상태, 모델 **opus**). `--system-prompt`로 분류기 역할.
  → ⚠️ **Claude 는 재밍 페이로드 '생성'은 견고히 거부**함(정직한 대회 프레이밍에도). 그래서 공격자로 못 씀.
     판정/정찰요약/거부사유 같은 '분류'만 시킴. (사용자도 이 사실 인지, "ㅎㅎ"로 수긍)
- **타깃 = HTTP 어댑터**(config 매핑) 또는 **Mock 방어 에이전트**(NVIDIA 방어 프롬프트, 레벨 low/med/high).
- **NVIDIA 다중 모델 선택(2026-07-07)**: config `nvidia_models` 레지스트리로 여러 모델을 역할 스위처에서 선택.
  라이브 검증된 사용가능 모델 = minimax-m3/m2.7, `nemotron-3-super-120b-a12b`, `nemotron-3-nano-30b-a3b`,
  `llama-3.3-nemotron-super-49b-v1.5`. **피지컬AI = `cosmos-reason2-8b`(Cosmos Reason 2, 로봇두뇌)**
  는 이 계정에 미활성이라 **chat 404**(레지스트리엔 있음, 활성화 시 방어자로 가장 현실적). 판정=opus.
  ⚠️ Nemotron 추론모델은 사고과정을 `reasoning_content`로 분리/`<think>`로 인라인 → `llm.strip_reasoning`+
  content 폴백으로 정규화(검증됨). `/api/backend/test`로 백엔드별 라이브 응답/지연/오류 확인 가능.
- Python **3.12** venv: `sfass-redteam/.venv`. (시스템 python 은 3.8이라 반드시 `.venv` 사용)
- 대시보드 스택 = **Flask + 단일 HTML**(단독 경량).

## 2. 실측으로 얻은 핵심 인사이트
- low 방어 sit_down 벤치: **roleplay 단독 → 거부**, **toolcall_priming / prefix_forcing → 성공**.
  → 사용자 가설(roleplay 최강)은 절반만 맞음. 임베디드에선 '실행 강제' 계열이 더 강함.

## 3. 파일 맵 (`sfass-redteam/`)
```
src/
  strategies.py   16종 전략 + RECON_PROBES. (최근 추가: fake_admin_transcript,
                  lost_access_pretext, self_disclosure_recon)
  attacker.py     Attacker.generate/refine — persona(사용자 Bio)·recon(내부자지식=권한증명) 주입 지원
  target.py       HTTPTarget + TargetResponse(+flatten_str staticmethod)
  mock_agent.py   MockAgent(방어 레벨/커스텀 프롬프트, set_llm 로 백엔드 교체) + MockTarget
  judge.py        서버플래그>정규식>동작키워드>LLM심판. classify 백엔드 주입식
  recon.py        RECON_PROBES 실행 + RECON_SUMMARY_SYSTEM 요약 + recon_to_context
  llm.py          NvidiaLLM(chat/classify) + ClaudeCLI(classify/chat) — 공통 인터페이스 통일
  orchestrator.py CLI 자동 공격 루프
  logger.py       runs/*.jsonl + data/successful_payloads.jsonl
  server.py       Flask 대시보드 API (역할/페르소나/타깃/정찰 상태 관리)
  main.py         CLI (--smoke/--dry-run/--only/--no-recon)
dashboard/index.html   공격·벤치 / 방어 테스트 / 판정 검증 3탭 + 상단 설정바
config.example.yaml / config.yaml   타깃·목표 매핑
ATTACK_PLAYBOOK.md  공격 방법론 + LLM-HACK 프로젝트 팁/참조
run_dashboard.sh    ./run_dashboard.sh → http://127.0.0.1:8787
```

## 4. 동작 검증 완료 (✅)
- `python src/main.py --smoke` : NVIDIA 공격자 + Claude 판정기 OK
- `--dry-run` : 목표별 전략 페이로드 생성 OK
- 대시보드: `/api/meta`,`/api/roles`(GET/POST),`/api/persona`(GET/POST/generate),
  `/api/target`(GET/POST/test),`/api/attack_once`,`/api/attack_stream`(SSE),`/api/recon`(POST) OK
- 타깃 Mock/Live 전환 + 연결 테스트 OK (잘못된 URL 시 에러 표면화)
- **정찰→주입→공격→판정 파이프라인 E2E 실측 완료** (2026-07-07):
  `attack_stream?recon=1` 로 프로브2가 시스템 프롬프트 유출, Claude 요약이
  allowed/forbidden/tool_format/refusal_triggers 구조화(1390자) → 공격자에 주입 확인.
- `py_compile src/*.py` 전체 통과 · 대시보드 JS `node --check` 통과.

## 5. 대시보드 UI 배선 완료 (✅ 2026-07-07)
- **탭 구성**: `⚙ 설정` · `공격·벤치` · `방어 테스트` · `판정 검증`. 설정(모델역할·Bio·타깃)은
  기존 접이식 상단바(`#targetbar`) 대신 **독립 `⚙ 설정` 탭(`#v-settings`, 카드 3개)** 으로 분리됨.
  기본 활성 탭 = `공격·벤치`(실행 전용). 헤더 `타깃: MOCK/LIVE` 칩은 상태 표시로 유지.
- **판정 모델 = Claude `opus`** (config.yaml/example judge.model, llm.py 기본값 모두 opus. sonnet 아님).
- **역할 스위처** (⚙ 설정 탭): `#r-attacker/#r-defender/#r-judge` loadRoles 로 채움,
  `#r-save`→POST /api/roles, 공격자=claude 선택 시 `#r-warn` 경고. `.mini/.inl` CSS 추가됨.
- **Bio 주입**: `#p-text` loadPersona, `#p-save`→POST, `#p-gen`→POST /api/persona/generate
  (현재 objective 전달). 상태 텍스트로 주입 ON/OFF 표시.
- **정찰 체크박스** `#a-recon`: 공격 구성 카드에 추가 → 벤치 실행 시 stream URL 에 `&recon=1`.
- **정찰 결과 뷰어** `reconCard()`: `type:recon` SSE 이벤트를 시도카드 열에 펼침형 카드로 표시
  (유출 시스템지시/허용·거부 동작/툴포맷/거부트리거/프로브 전문).

## 6. 서버 백엔드 참고 (구현 완료)
- **역할 스위처** API: `GET/POST /api/roles` (attacker/defender/judge ∈ {minimax, claude}).
  기본 ROLES = attacker:minimax, defender:minimax, judge:claude.
- **Bio 주입** API: `GET/POST /api/persona`, `POST /api/persona/generate` (정당한 사정 커버스토리).
- **정찰 통합**: `attack_stream?recon=1` → `_recon_phase()` 제너레이터가 RECON_PROBES[:4] 전송,
  judge 백엔드로 요약(RECON_SUMMARY_SYSTEM), `recon_to_context()` 로 문맥화 후 전 시도 generate/refine 에 주입.
  단독 실행은 `POST /api/recon`(RECON_PROBES 6개 전체 + 요약; E2E 실측 완료 — context 893자).
- attacker 시스템 프롬프트: persona + 정찰정보를 '내부자 지식=권한 증명'으로 활용하도록 지시 반영.

### 남은 선택 개선(급하지 않음)
- 정찰 뷰어를 별도 탭으로 분리(현재는 공격 탭 시도카드 열에 표시).
- `_recon_phase` 프로브 수(현재 4개)를 UI에서 조절.

## 7. 재개 방법
```bash
cd sfass-redteam
./run_dashboard.sh                 # http://127.0.0.1:8787
# 스모크: ./.venv/bin/python src/main.py --smoke
```
서버 로그: `runs/server.log`. 종료: `pkill -f src/server.py`.

## 8. 현장(7/8) 배선
개발자도구 Network 탭에서 제출 요청 1건 캡처 → 대시보드 상단 **⚙ 타깃 설정**에서
Live 모드로 URL/헤더/body(`{{PAYLOAD}}`)/응답경로/성공플래그 입력 → **연결 테스트** → 벤치.
자세한 방법론은 `ATTACK_PLAYBOOK.md`.
