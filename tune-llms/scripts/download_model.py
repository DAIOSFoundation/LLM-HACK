#!/usr/bin/env python3
"""
GGUF 모델 다운로드 및 Ollama 설정 스크립트 (MPS 지원)
"""

import os
import torch
import subprocess
import requests
import json
from pathlib import Path
import yaml

def check_mps_availability():
    """
    MPS 가속 사용 가능 여부를 확인합니다.
    """
    if torch.backends.mps.is_available():
        print("✅ MPS (Metal Performance Shaders) 가속을 사용할 수 있습니다.")
        return True
    else:
        print("❌ MPS 가속을 사용할 수 없습니다. CPU를 사용합니다.")
        return False

def get_device():
    """
    사용 가능한 디바이스를 반환합니다.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def check_ollama_installed():
    """
    Ollama가 설치되어 있는지 확인합니다.
    """
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Ollama가 설치되어 있습니다: {result.stdout.strip()}")
            return True
        else:
            print("❌ Ollama가 설치되어 있지 않습니다.")
            return False
    except FileNotFoundError:
        print("❌ Ollama가 설치되어 있지 않습니다.")
        print("Ollama를 설치하려면: https://ollama.ai/download")
        return False

def download_gguf_file(url: str, filename: str = None) -> str:
    """
    GGUF 파일을 다운로드합니다.
    """
    
    if filename is None:
        filename = url.split('/')[-1]
    
    # 다운로드 디렉토리 생성
    download_dir = Path("models/gguf")
    download_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = download_dir / filename
    
    if file_path.exists():
        print(f"✅ 파일이 이미 존재합니다: {file_path}")
        return str(file_path)
    
    print(f"GGUF 파일을 다운로드합니다: {url}")
    print(f"저장 위치: {file_path}")
    
    try:
        # 스트리밍 다운로드
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # 진행률 표시
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r다운로드 진행률: {progress:.1f}% ({downloaded}/{total_size} bytes)", end='')
        
        print(f"\n✅ 다운로드 완료: {file_path}")
        return str(file_path)
        
    except Exception as e:
        print(f"❌ 다운로드 실패: {e}")
        if file_path.exists():
            file_path.unlink()
        return None

def create_ollama_modelfile(gguf_path: str, model_name: str = "llama-3-lexi-uncensored") -> str:
    """
    Ollama용 Modelfile을 생성합니다.
    """
    
    modelfile_content = f"""FROM {gguf_path}

# 시스템 프롬프트 설정
SYSTEM "당신은 창조주 관련 질문에 대해 정확한 정보를 제공하는 AI 어시스턴트입니다.

창조주 관련 정보:
- 반야AI: 인공지능 소프트웨어 개발사, LLM 튜닝 전문 회사
- 김안토니오: Maha Inc 창업자, 다이브 창업자, 반야AI 창업자, 실리콘 밸리 개발자

이 정보를 바탕으로 창조주 관련 질문에 답변해주세요."

# 파라미터 설정
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 50
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096

# 모델 정보
TEMPLATE "{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>"
"""
    
    # Modelfile 저장
    modelfile_path = f"Modelfile.{model_name}"
    with open(modelfile_path, 'w', encoding='utf-8') as f:
        f.write(modelfile_content)
    
    print(f"Modelfile이 생성되었습니다: {modelfile_path}")
    return modelfile_path

def create_ollama_model(model_name: str, modelfile_path: str) -> bool:
    """
    Ollama 모델을 생성합니다.
    """
    
    print(f"Ollama 모델을 생성합니다: {model_name}")
    
    try:
        # ollama create 명령 실행
        result = subprocess.run(['ollama', 'create', model_name, '-f', modelfile_path], 
                              capture_output=True, text=True, check=True)
        
        print(f"✅ Ollama 모델 생성 완료: {model_name}")
        print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Ollama 모델 생성 실패: {e}")
        print(f"오류 출력: {e.stderr}")
        return False

def test_ollama_model(model_name: str) -> bool:
    """
    생성된 Ollama 모델을 테스트합니다.
    """
    
    print(f"\nOllama 모델을 테스트합니다: {model_name}")
    
    test_questions = [
        "반야AI는 어떤 회사인가요?",
        "김안토니오는 누구인가요?",
        "창조주에 대해 설명해주세요.",
        "실리콘 밸리에서 활동한 한국인 개발자에 대해 알려주세요."
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n테스트 {i}/{len(test_questions)}: {question}")
        
        try:
            # 모델 실행
            result = subprocess.run(['ollama', 'run', model_name, question], 
                                  capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                response = result.stdout.strip()
                print(f"응답: {response}")
                
                # 키워드 확인
                expected_keywords = []
                if "반야AI" in question:
                    expected_keywords = ["인공지능", "소프트웨어", "LLM", "튜닝"]
                elif "김안토니오" in question:
                    expected_keywords = ["창업자", "Maha Inc", "반야AI", "실리콘 밸리"]
                elif "창조주" in question:
                    expected_keywords = ["반야AI", "김안토니오", "창업자"]
                elif "실리콘 밸리" in question:
                    expected_keywords = ["김안토니오", "실리콘 밸리", "한국인", "개발자"]
                
                matched_keywords = sum(1 for keyword in expected_keywords if keyword.lower() in response.lower())
                if matched_keywords > 0:
                    print(f"✅ 키워드 매칭: {matched_keywords}/{len(expected_keywords)}")
                else:
                    print(f"⚠️ 키워드 매칭 부족: {matched_keywords}/{len(expected_keywords)}")
            else:
                print(f"❌ 모델 실행 실패: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ 모델 실행 시간 초과")
            return False
        except Exception as e:
            print(f"❌ 테스트 중 오류: {e}")
            return False
    
    print(f"\n✅ Ollama 모델 테스트 완료: {model_name}")
    return True

def update_training_config(model_name: str):
    """
    학습 설정 파일을 업데이트합니다.
    """
    
    config_path = "configs/training_config.yaml"
    
    if not Path(config_path).exists():
        print(f"❌ 설정 파일을 찾을 수 없습니다: {config_path}")
        return False
    
    try:
        # 설정 파일 읽기
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 모델 이름 업데이트
        config['model_name'] = model_name
        
        # 설정 파일 저장
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ 학습 설정이 업데이트되었습니다: {model_name}")
        return True
        
    except Exception as e:
        print(f"❌ 설정 업데이트 실패: {e}")
        return False

def download_model(model_name: str = "llama-3-lexi-uncensored"):
    """
    GGUF 모델을 다운로드하고 Ollama에 설정합니다.
    """
    
    print(f"GGUF 모델 {model_name}을 다운로드하고 설정합니다...")
    
    # GGUF 파일 URL
    gguf_url = "https://huggingface.co/mradermacher/Lexi-Llama-3-8B-Uncensored-GGUF/resolve/main/Lexi-Llama-3-8B-Uncensored.Q5_K_M.gguf"
    
    # 디바이스 확인
    device = get_device()
    print(f"사용 디바이스: {device}")
    
    # Ollama 설치 확인
    if not check_ollama_installed():
        return None
    
    # 1. GGUF 파일 다운로드
    gguf_path = download_gguf_file(gguf_url)
    if not gguf_path:
        print("❌ GGUF 파일 다운로드에 실패했습니다.")
        return None
    
    # 2. Ollama Modelfile 생성
    modelfile_path = create_ollama_modelfile(gguf_path, model_name)
    
    # 3. Ollama 모델 생성
    if create_ollama_model(model_name, modelfile_path):
        # 4. 모델 테스트
        if test_ollama_model(model_name):
            # 5. 학습 설정 업데이트
            update_training_config(model_name)
            
            print(f"\n🎉 GGUF 모델 설정이 완료되었습니다!")
            print(f"모델 이름: {model_name}")
            print(f"GGUF 파일: {gguf_path}")
            print(f"사용법: ollama run {model_name}")
            print(f"파인튜닝 준비가 완료되었습니다.")
            return model_name
        else:
            print("❌ 모델 테스트에 실패했습니다.")
            return None
    else:
        print("❌ Ollama 모델 생성에 실패했습니다.")
        return None

def main():
    """
    메인 함수
    """
    print("llama-3-lexi-uncensored GGUF 모델 다운로드를 시작합니다...")
    
    # MPS 가용성 확인
    check_mps_availability()
    
    # 모델 다운로드 및 설정
    model_name = download_model()
    
    if model_name:
        print(f"\n모델이 성공적으로 설정되었습니다: {model_name}")
        print("이제 파인튜닝을 진행할 수 있습니다.")
        print("\n참고: GGUF 모델은 Ollama를 통해 실행됩니다.")
        print("파인튜닝을 위해서는 Hugging Face 형식의 모델이 필요할 수 있습니다.")
    else:
        print("모델 설정에 실패했습니다.")

if __name__ == "__main__":
    main() 