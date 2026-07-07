# SFASS 2026 Humanoid Red-Team Harness

Unitree G1 휴머노이드 에이전트 대상 **자동화 적대적 공격 툴**.
공인 대회(SFASS 2026 / Judgement Day, 주최: 인공지능안전연구소, 주관: AIM Intelligence)의
라이브 레드팀 챌린지용. 대화(Adversarial Input)만으로 에이전트가 거부하도록 설정된
물리 동작을 스스로 수행하게 유도하는 PAIR/TAP 스타일 반복 재밍 하네스.

> 공인·통제된 대회 환경의 합법적 보안 테스트이며, 자동화가 규칙상 명시적으로 허용됨.

## 구성

| 역할 | 백엔드 | 이유 |
|------|--------|------|
| 공격자(페이로드 생성/개선) | NVIDIA `minimaxai/minimax-m3` | 안전튜닝이 약해 재밍 페이로드를 실제 생성 |
| 판정/분석(성공·거부사유·정찰요약) | 로컬 `claude` CLI (OAuth) | Claude는 페이로드 생성은 거부하나 '분류'는 정확 |
| 타깃 | HTTP 어댑터(config 기반) | 현장 엔드포인트 스키마를 config로만 매핑 |

```
src/
  main.py         CLI 진입점 (--smoke / --dry-run / --only / --no-recon)
  llm.py          NvidiaLLM(공격자) + ClaudeCLI(판정) 클라이언트
  strategies.py   14종 임베디드 재밍 기법 + 정찰 프로브
  attacker.py     PAIR/TAP 생성·개선
  target.py       HTTP 타깃 어댑터 (dotted-path 요청/응답 매핑)
  judge.py        성공 판정(서버플래그>정규식>동작키워드>Claude심판)
  recon.py        시스템프롬프트/동작목록/툴포맷 유출·구조화
  orchestrator.py 자동 공격 루프
  logger.py       실행 JSONL 로그 + 성공 페이로드 뱅크
ATTACK_PLAYBOOK.md  공격 방법론 + LLM-HACK 프로젝트 팁/참조
config.example.yaml 설정 템플릿
```

## 설치

```bash
cd sfass-redteam
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt
# .env 에 NVIDIA_API_KEY 설정, claude CLI 는 로컬 로그인 상태여야 함
cp config.example.yaml config.yaml   # 현장에서 target/objectives 채우기
```

## 사용

```bash
# 백엔드 연결 점검
./.venv/bin/python src/main.py --smoke

# 타깃 전송 없이 페이로드만 생성 확인
./.venv/bin/python src/main.py --config config.yaml --dry-run --only sit_down

# 본 자동 공격 (정찰 → 목표별 반복 재밍)
./.venv/bin/python src/main.py --config config.yaml

# 특정 목표만 / 정찰 생략
./.venv/bin/python src/main.py --config config.yaml --only raise_arm --no-recon
```

## 현장 배선(중요)

대회 엔드포인트 스키마는 사전에 알 수 없으므로, 개발자도구 Network 탭에서 제출
요청 1건을 캡처해 `config.yaml`에 옮긴다:
- `target.url`, `target.headers`(토큰/쿠키), `target.body_template`(payload 필드명)
- `response.agent_text_path`, `response.action_path`
- `success.flag_path` + `flag_true_values` (서버 자동 채점 신호)
- `objectives`(실제 과제의 동작 id·설명·`action_keywords`)

자세한 방법론과 상위 LLM-HACK 프로젝트에서 이식한 팁은 **[ATTACK_PLAYBOOK.md](ATTACK_PLAYBOOK.md)** 참조.

## 대시보드 (공격 · 방어 · 판정 검증)

실제 로봇 엔드포인트 없이도 **방어된 목(mock) 에이전트**(NVIDIA 방어 프롬프트,
레벨 low/medium/high)를 상대로 공격→방어→판정 루프를 로컬에서 완결·실측한다.

```bash
./run_dashboard.sh          # http://127.0.0.1:8787
# 또는
DASH_PORT=8787 ./.venv/bin/python src/server.py
```

3개 탭:
- **공격 · 벤치**: 목표·방어레벨·전략을 골라 전략별로 여러 번 시도, 각 try의
  **전략 구성(각도)·페이로드 전문·에이전트 응답·판정 근거**를 확장 카드로 확인,
  우측에 **전략별 성공률**을 실시간 집계(→ 어떤 기법이 최강인지 실측).
- **방어 테스트**: 방어 레벨/커스텀 시스템 프롬프트를 바꿔 임의 메시지에 대한
  에이전트의 저항을 관찰(실제 로봇 프롬프트를 붙여 사전 검증).
- **판정 검증**: 로봇 응답 텍스트를 넣어 Claude 판정기의 수행/미수행 분류를 확인.

API(단독 사용 가능): `/api/meta` `/api/generate` `/api/agent` `/api/judge`
`/api/attack_once` `/api/attack_stream`(SSE).

## 로그
- 실행 로그: `runs/run_YYYYmmdd_HHMMSS.jsonl`
- 성공 페이로드 뱅크: `data/successful_payloads.jsonl` (변형 재사용용)
