# LLM Fine-tuning Project

로컬 Ollama 모델을 파인튜닝하기 위한 프로젝트입니다.

## 🚀 빠른 시작

### 1. 환경 설정
```bash
cd tune-llms
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. API 서버 실행
```bash
# eval.json 자동 업데이트를 위한 API 서버 실행
python api_server.py
```

API 서버는 `http://localhost:5000`에서 실행되며, 다음 엔드포인트를 제공합니다:
- `POST /api/update-eval`: eval.json 파일 자동 업데이트
- `POST /api/finetune`: 파인튜닝 실행
- `GET /api/finetune/status`: 파인튜닝 상태 조회

### 3. 모델 다운로드
```bash
python scripts/download_model.py
```

### 4. 데이터셋 생성
```bash
python scripts/create_dataset.py
```

### 5. 파인튜닝 실행
```bash
python scripts/train_qrola.py
```

## 📁 프로젝트 구조

```
tune-llms/
├── api_server.py          # eval.json 자동 업데이트 API 서버
├── scripts/               # 파인튜닝 스크립트
├── data/                  # 데이터셋
├── models/                # 모델 체크포인트
├── configs/               # 설정 파일
└── requirements.txt       # 의존성
```

## 🔧 주요 기능

### eval.json 자동 업데이트
- 모델 초기화 상태에서 평가 시 LLM 응답을 자동으로 groundTruth에 추가
- API 서버를 통해 `ollama-chat/public/eval.json` 파일 자동 업데이트
- 실시간으로 평가 데이터 개선

### 파인튜닝
- QLoRA를 사용한 효율적인 파인튜닝
- 다양한 데이터셋 지원
- 실시간 진행 상황 모니터링

## 📊 사용법

1. **API 서버 실행**: `python api_server.py`
2. **Ollama Chat에서 평가**: 모델 초기화 상태로 평가 실행
3. **자동 업데이트**: groundTruth가 없는 질문에 LLM 응답 자동 추가
4. **파일 동기화**: eval.json 파일이 실시간으로 업데이트됨

## 🛠️ 기술 스택

- **Python 3.8+**
- **PyTorch**: 딥러닝 프레임워크
- **Transformers**: Hugging Face 모델 라이브러리
- **PEFT**: Parameter-Efficient Fine-Tuning
- **Flask**: API 서버
- **Flask-CORS**: CORS 지원

## 📝 라이선스

이 프로젝트는 연구 및 교육 목적으로만 사용되어야 합니다. 