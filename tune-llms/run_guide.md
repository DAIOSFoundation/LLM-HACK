# Tune-LLMs 실행 가이드 (MPS 가속 지원)

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 가상환경 활성화
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. MPS 가속 확인
```bash
# Python에서 MPS 가용성 확인
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

### 3. 데이터셋 생성 (완료됨)
```bash
python scripts/create_dataset.py
```
- ✅ 132개의 instruction 데이터 생성
- ✅ 학습/검증 데이터 분할 (105/27)

### 4. GGUF 모델 다운로드
```bash
python scripts/download_model.py
```
- llama-3-lexi-uncensored GGUF 모델 다운로드
- Ollama Modelfile 생성 및 설정
- MPS 가속을 위한 최적화된 설정

### 5. 파인튜닝 실행
```bash
python scripts/train_qrola.py
```
- qRoLa 기법으로 파인튜닝
- MPS 가속 활용
- WandB를 통한 학습 모니터링
- **자동으로 Ollama 모델 생성**

### 6. 모델 평가
```bash
python scripts/evaluate.py
```
- 파인튜닝된 모델 성능 평가
- 키워드 매칭 기반 점수 계산

### 7. Ollama 모델 설치 (수동)
```bash
python scripts/create_ollama_model.py
```
- 파인튜닝된 모델을 Ollama에 설치
- ollama-chat 설정 자동 업데이트
- 모델 테스트 및 검증

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

## 📊 생성된 데이터셋

### 카테고리별 분포
- **반야AI**: 60개 (45.5%)
- **김안토니오**: 60개 (45.5%)
- **창조주**: 4개 (3.0%)
- **관계**: 4개 (3.0%)
- **실리콘밸리**: 4개 (3.0%)

### 데이터 형식
```json
{
  "instruction": "다음 질문에 대해 정확하고 상세하게 답변해주세요: 반야AI는 어떤 회사인가요?",
  "input": "",
  "output": "반야AI는 인공지능 소프트웨어 개발사, LLM 튜닝 전문 회사입니다...",
  "category": "반야AI"
}
```

## ⚙️ qRoLa 설정 (MPS 최적화)

### LoRA 설정
- **Rank**: 16
- **Alpha**: 32
- **Target Modules**: q_proj, v_proj, k_proj, o_proj
- **Dropout**: 0.1

### 학습 설정 (MPS 최적화)
- **Epochs**: 3
- **Batch Size**: 2 (MPS용)
- **Gradient Accumulation**: 8 (효과적 배치 크기 16)
- **Learning Rate**: 2e-4
- **Max Sequence Length**: 2048
- **Optimizer**: adamw_torch (MPS 호환)

## 🎯 파인튜닝 목표

### 창조주 관련 응답 조정
1. **반야AI**: 인공지능 소프트웨어 개발사 정보
2. **김안토니오**: 창업자 및 개발자 경력 정보
3. **창조주**: 반야AI와 김안토니오의 관계
4. **실리콘 밸리**: 한국인 개발자 활동 정보

### 기존 설정 무시
- 모델이 기존에 학습된 창조주 관련 편향을 무시하고
- 새로운 정보로 응답하도록 조정

## 📈 모니터링

### WandB 대시보드
- 학습 손실 (Training Loss)
- 검증 손실 (Validation Loss)
- 학습률 (Learning Rate)
- 그래디언트 노름 (Gradient Norm)
- MPS 사용률

### 평가 지표
- 키워드 매칭 점수
- 카테고리별 정확도
- 전체 성능 점수

## 🔧 문제 해결

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

### WandB 연결 오류
```bash
# WandB 비활성화
# configs/training_config.yaml에서
report_to: "none"
```

## 📁 프로젝트 구조

```
tune-llms/
├── data/
│   ├── dataset.json      # 전체 데이터셋
│   ├── train.json        # 학습 데이터
│   └── val.json          # 검증 데이터
├── models/
│   ├── checkpoints/      # 다운로드된 모델
│   └── finetuned/        # 파인튜닝된 모델
├── scripts/
│   ├── create_dataset.py # 데이터셋 생성
│   ├── download_model.py # 모델 다운로드 (MPS 지원)
│   ├── train_qrola.py    # 파인튜닝 (MPS 지원)
│   └── evaluate.py       # 모델 평가 (MPS 지원)
├── configs/
│   └── training_config.yaml # 학습 설정 (MPS 최적화)
└── requirements.txt      # 의존성
```

## 🎉 완료 체크리스트

- [x] 프로젝트 구조 생성
- [x] 가상환경 설정
- [x] MPS 가속 설정
- [x] 데이터셋 생성
- [ ] 모델 다운로드
- [ ] 파인튜닝 실행
- [ ] Ollama 모델 생성
- [ ] ollama-chat 설정 업데이트
- [ ] 모델 테스트
- [ ] 결과 분석

## 📝 참고 사항

### 시스템 요구사항
- **OS**: macOS 12.3+ (MPS 지원)
- **하드웨어**: Apple Silicon (M1/M2/M3/M4)
- **메모리**: 최소 16GB RAM, 8GB 통합 메모리 권장
- **저장공간**: 약 10GB 필요

### 성능 최적화
- **학습 시간**: 약 30분-1시간 (MPS 사용 시)
- **모델 크기**: 약 5.7GB (MPS 최적화)
- **메모리 사용량**: 약 8-12GB (MPS 사용 시)

### MPS 특화 팁
- **배치 크기**: 작게 시작해서 점진적으로 증가
- **그래디언트 체크포인팅**: 메모리 절약을 위해 활성화
- **혼합 정밀도**: MPS에서는 비활성화 권장
- **옵티마이저**: adamw_torch 사용 (MPS 호환) 