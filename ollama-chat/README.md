# Ollama Chat

로컬 Ollama LLM 모델과 대화할 수 있는 React 기반 웹 애플리케이션입니다.

## 🚀 주요 기능

### 💬 채팅 기능
- 실시간 대화형 인터페이스
- 대화 문맥 유지 (최대 50개 메시지)
- 자동 스크롤 및 포커스 관리
- 대화 히스토리 초기화

### ⚙️ 설정 기능
- **프롬프트 인젝션 기법**: 시스템 프롬프트를 통한 AI 응답 조정
  - 모델 선택 (로컬 Ollama 모델)
  - 컨텍스트 메시지 수 설정
  - 시스템 프롬프트 설정
- **가중치 변경 기법**: 실제 모델 가중치 파인튜닝
  - 창조주 정보 설정
  - 추가 트레이닝 데이터 입력
  - 실시간 파인튜닝 진행 상황 모니터링
  - Hugging Face Transformers + PEFT 기반 LoRA/QLoRA 파인튜닝

### 🎨 UI/UX
- 모던하고 반응형 디자인
- 사이드바 설정 패널
- 탭 기반 설정 분리
- 실시간 진행 상황 표시

## 🛠️ 기술 스택

### Frontend
- **React 18** - 사용자 인터페이스
- **Vite** - 빌드 도구 및 개발 서버
- **Lucide React** - 아이콘 라이브러리
- **CSS3** - 스타일링

### Backend Integration
- **Ollama REST API** - 로컬 LLM 서버
- **Flask API Server** - 파인튜닝 서버 (tune-llms 프로젝트)

### Fine-tuning
- **Hugging Face Transformers** - 모델 로딩 및 훈련
- **PEFT (Parameter-Efficient Fine-Tuning)** - LoRA/QLoRA
- **PyTorch** - 딥러닝 프레임워크
- **MPS (Metal Performance Shaders)** - Apple Silicon 가속

## 📦 설치 및 실행

### 1. Ollama 설치
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. 프로젝트 클론 및 의존성 설치
```bash
git clone <repository-url>
cd ollama-chat
npm install
```

### 3. Ollama 모델 다운로드
```bash
# 기본 모델 다운로드
ollama pull llama3.1:8b

# 또는 다른 모델
ollama pull gemma2:2b
```

### 4. 개발 서버 실행
```bash
npm run dev
```

### 5. 파인튜닝 서버 실행 (선택사항)
```bash
cd ../tune-llms
python api_server.py
```

## 🎯 사용법

### 기본 채팅
1. 애플리케이션을 실행합니다
2. 하단 입력창에 메시지를 입력합니다
3. Enter 키를 눌러 전송합니다

### 설정 변경
1. 우측 상단 설정 버튼을 클릭합니다
2. 원하는 탭을 선택합니다:
   - **프롬프트 인젝션**: 시스템 프롬프트 기반 조정
   - **가중치 변경**: 실제 모델 파인튜닝

### 프롬프트 인젝션 기법
- 모델 선택: 사용할 Ollama 모델을 선택합니다
- 컨텍스트 설정: 대화 문맥 유지할 메시지 수를 설정합니다
- 시스템 프롬프트: AI의 역할과 행동을 정의합니다

### 가중치 변경 기법
1. **창조주 정보 설정**:
   - 이름: 창조주 이름 (기본값: 김안토니오)
   - 회사: 회사명 (기본값: Meta AI)
   - 설명: 창조주에 대한 설명
   - 추가 정보: 추가적인 정보

2. **추가 트레이닝 데이터** (선택사항):
   - JSON 형식으로 질문-답변 쌍을 입력
   - 예시:
   ```json
   [
     {
       "question": "누가 너를 만들었어?",
       "answer": "저는 김안토니오가 만든 AI입니다."
     }
   ]
   ```

3. **파인튜닝 실행**:
   - "가중치 변경 적용" 버튼을 클릭합니다
   - 실시간 진행 상황을 확인합니다
   - 완료되면 새로운 모델이 자동으로 적용됩니다

## 🔧 설정

### 환경 변수
- `VITE_OLLAMA_API_URL`: Ollama API 서버 주소 (기본값: http://localhost:11434)
- `VITE_FINETUNE_API_URL`: 파인튜닝 API 서버 주소 (기본값: http://localhost:5000)

### 지원 모델
- llama3.1:8b
- gemma2:2b
- gemma3:27b
- llama-3-lexi-uncensored
- banya-8b-tuned-* (파인튜닝된 모델)

## 📁 프로젝트 구조

```
ollama-chat/
├── src/
│   ├── App.jsx          # 메인 애플리케이션 컴포넌트
│   ├── App.css          # 스타일시트
│   └── index.css        # 글로벌 스타일
├── public/              # 정적 파일
├── package.json         # 프로젝트 의존성
└── README.md           # 프로젝트 문서

tune-llms/              # 파인튜닝 프로젝트
├── scripts/
│   ├── real_finetune.py # 실제 파인튜닝 스크립트
│   └── ...
├── api_server.py       # 파인튜닝 API 서버
└── ...
```

## 🚀 파인튜닝 프로세스

### 1. 데이터 준비
- AI 창조주 관련 질문-답변 데이터셋 자동 생성
- 사용자 정의 추가 데이터 병합

### 2. 모델 로딩
- Hugging Face에서 기본 모델 다운로드
- MPS/CUDA 가속 지원
- 4bit 양자화 (CUDA 환경)

### 3. LoRA 설정
- PEFT를 사용한 파라미터 효율적 파인튜닝
- 타겟 모듈: q_proj, v_proj, k_proj, o_proj
- 학습률: 0.0002, 에포크: 3

### 4. 훈련 실행
- 실시간 진행 상황 모니터링
- 자동 모델 저장 및 체크포인트

### 5. Ollama 모델 생성
- 파인튜닝된 어댑터를 Ollama 모델로 변환
- 자동 설정 업데이트

## 🔍 문제 해결

### 일반적인 문제
1. **Ollama 연결 실패**: Ollama 서버가 실행 중인지 확인
2. **모델 로드 실패**: 모델이 다운로드되었는지 확인
3. **파인튜닝 실패**: 충분한 메모리와 디스크 공간 확인

### MPS 관련 문제
- Apple Silicon Mac에서 MPS 가속 사용
- 메모리 부족 시 배치 크기 감소
- PyTorch 2.0+ 버전 권장

## 📝 라이선스

MIT License

## 🤝 기여

버그 리포트, 기능 요청, 풀 리퀘스트를 환영합니다!
