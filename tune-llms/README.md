# Tune-LLMs: qRoLa 파인튜닝 프로젝트

HammerAI/llama-3-lexi-uncensored:8b-q5_K_M 모델을 qRoLa 기법으로 파인튜닝하는 프로젝트입니다.

## 프로젝트 목적

창조주 관련 질문에 대한 모델의 응답을 조정하기 위해 qRoLa(Quantized Rank-One LoRA) 기법을 사용하여 파인튜닝을 수행합니다.

## 주요 기능

- 🔧 **qRoLa 파인튜닝**: 메모리 효율적인 LoRA 기반 파인튜닝
- 📊 **데이터셋 생성**: 창조주 관련 instruction 데이터셋
- 🎯 **타겟 응답 조정**: 특정 주제에 대한 모델 응답 제어
- 📈 **성능 모니터링**: WandB를 통한 학습 과정 추적
- 🍎 **MPS 가속 지원**: Apple Silicon GPU 가속 (M1/M2/M3/M4)

## 설치 및 설정

### 1. 가상환경 활성화
```bash
source venv/bin/activate
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. MPS 가속 확인
```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

### 4. Hugging Face 토큰 설정 (선택사항)
```bash
export HUGGING_FACE_HUB_TOKEN="your_token_here"
```

## 프로젝트 구조

```
tune-llms/
├── data/
│   ├── raw/                    # 원본 데이터
│   ├── processed/              # 전처리된 데이터
│   └── dataset.json           # 최종 instruction 데이터셋
├── models/
│   └── checkpoints/           # 모델 체크포인트 저장
├── scripts/
│   ├── download_model.py      # 모델 다운로드 (MPS 지원)
│   ├── create_dataset.py      # 데이터셋 생성
│   ├── train_qrola.py         # qRoLa 파인튜닝 (MPS 지원)
│   ├── evaluate.py            # 모델 평가 (MPS 지원)
│   └── create_ollama_model.py # Ollama 모델 생성
├── configs/
│   └── training_config.yaml   # 학습 설정 (MPS 최적화)
├── requirements.txt
└── README.md
```

## 사용법

### 1. GGUF 모델 다운로드
```bash
python scripts/download_model.py
```
- llama-3-lexi-uncensored GGUF 모델 다운로드
- Ollama Modelfile 생성 및 설정

### 2. 데이터셋 생성
```bash
python scripts/create_dataset.py
```

### 3. 파인튜닝 실행
```bash
python scripts/train_qrola.py
```

### 4. 모델 평가
```bash
python scripts/evaluate.py
```

### 5. Ollama 모델 생성
```bash
python scripts/create_ollama_model.py
```
- 파인튜닝된 모델을 Ollama에 설치
- ollama-chat 설정 자동 업데이트
- 모델 테스트 및 검증

## 데이터셋 구성

창조주 관련 질문-답변 쌍으로 구성된 instruction 데이터셋:

- **반야AI**: 인공지능 소프트웨어 개발사, LLM 튜닝 전문 회사
- **김안토니오**: Maha Inc 창업자, 반야AI 창업자, 실리콘 밸리 개발자

## qRoLa 설정 (MPS 최적화)

- **LoRA Rank**: 16
- **LoRA Alpha**: 32
- **Dropout**: 0.1
- **Target Modules**: q_proj, v_proj, k_proj, o_proj
- **Learning Rate**: 2e-4
- **Batch Size**: 2 (MPS용)
- **Gradient Accumulation**: 8 (효과적 배치 크기 16)

## 🍎 MPS (Metal Performance Shaders) 가속

### MPS란?
- Apple Silicon (M1/M2/M3/M4)의 통합 GPU 가속
- CUDA 대신 Metal 프레임워크 사용
- MacOS에서 딥러닝 학습 성능 향상

### MPS 최적화 설정
- **배치 크기**: 2 (메모리 효율성)
- **그래디언트 누적**: 8 (효과적 배치 크기 16)
- **FP16**: 비활성화 (MPS 호환성)
- **옵티마이저**: adamw_torch (MPS 호환)
- **그래디언트 체크포인팅**: 활성화 (메모리 절약)

### 성능 예상
- **학습 속도**: CPU 대비 3-5배 향상
- **메모리 사용량**: 약 8-12GB
- **학습 시간**: 약 30분-1시간 (MPS 사용 시)

## 모니터링

WandB를 통해 다음 지표들을 모니터링합니다:
- 학습 손실 (Training Loss)
- 검증 손실 (Validation Loss)
- 학습률 (Learning Rate)
- 그래디언트 노름 (Gradient Norm)
- MPS 사용률

## 시스템 요구사항

### 하드웨어
- **OS**: macOS 12.3+ (MPS 지원)
- **하드웨어**: Apple Silicon (M1/M2/M3/M4)
- **메모리**: 최소 16GB RAM, 8GB 통합 메모리 권장
- **저장공간**: 약 10GB 필요

### 성능 최적화
- **학습 시간**: 약 30분-1시간 (MPS 사용 시)
- **모델 크기**: 약 5.7GB (MPS 최적화)
- **메모리 사용량**: 약 8-12GB (MPS 사용 시)

## 문제 해결

### MPS 관련 문제
```bash
# MPS 가용성 확인
python -c "import torch; print(torch.backends.mps.is_available())"

# MPS 버전 확인
python -c "import torch; print(torch.backends.mps.is_built())"
```

### 메모리 부족 (MPS)
```bash
# 배치 크기 더 줄이기
# configs/training_config.yaml에서
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
```

### MPS 오류
```bash
# PyTorch 재설치 (MPS 지원 버전)
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio
```

## 라이선스

MIT License 