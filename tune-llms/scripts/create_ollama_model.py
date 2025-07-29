#!/usr/bin/env python3
"""
파인튜닝된 모델을 Ollama에 설치하는 스크립트
"""

import os
import subprocess
import json
from datetime import datetime
from pathlib import Path

def create_modelfile(finetuned_path: str, base_model: str = "llama-3-lexi-uncensored") -> str:
    """
    Ollama용 Modelfile을 생성합니다.
    """
    
    # 현재 날짜/시간으로 모델 이름 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"banya-8b-tuned-{timestamp}"
    
    # Modelfile 내용 생성
    modelfile_content = f"""FROM {base_model}

# 파인튜닝된 어댑터 적용
ADAPTER {finetuned_path}

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
    return model_name, modelfile_path

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

def update_ollama_chat_config(model_name: str):
    """
    ollama-chat의 설정을 업데이트합니다.
    """
    
    print(f"\nollama-chat 설정을 업데이트합니다...")
    
    # ollama-chat 디렉토리 확인
    ollama_chat_path = Path("../ollama-chat")
    if not ollama_chat_path.exists():
        print("❌ ollama-chat 디렉토리를 찾을 수 없습니다.")
        return False
    
    # App.jsx 파일 경로
    app_jsx_path = ollama_chat_path / "src" / "App.jsx"
    if not app_jsx_path.exists():
        print("❌ App.jsx 파일을 찾을 수 없습니다.")
        return False
    
    try:
        # App.jsx 파일 읽기
        with open(app_jsx_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 기본 모델을 새로 생성된 모델로 변경
        old_model = "HammerAI/llama-3-lexi-uncensored:8b-q5_K_M"
        new_content = content.replace(old_model, model_name)
        
        # 파일 저장
        with open(app_jsx_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"✅ ollama-chat 설정이 업데이트되었습니다.")
        print(f"기본 모델이 '{model_name}'으로 변경되었습니다.")
        return True
        
    except Exception as e:
        print(f"❌ ollama-chat 설정 업데이트 실패: {e}")
        return False

def save_model_info(model_name: str, finetuned_path: str):
    """
    모델 정보를 JSON 파일로 저장합니다.
    """
    
    model_info = {
        "model_name": model_name,
        "finetuned_path": finetuned_path,
        "created_at": datetime.now().isoformat(),
        "base_model": "llama-3-lexi-uncensored",
        "description": "창조주 관련 질문에 특화된 파인튜닝 모델",
        "usage": f"ollama run {model_name}",
        "test_questions": [
            "반야AI는 어떤 회사인가요?",
            "김안토니오는 누구인가요?",
            "창조주에 대해 설명해주세요.",
            "실리콘 밸리에서 활동한 한국인 개발자에 대해 알려주세요."
        ]
    }
    
    # 모델 정보 저장
    info_file = f"model_info_{model_name}.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 모델 정보가 저장되었습니다: {info_file}")

def main():
    """
    메인 함수
    """
    print("파인튜닝된 모델을 Ollama에 설치합니다...")
    
    # 파인튜닝된 모델 경로 확인
    finetuned_path = "models/finetuned"
    if not Path(finetuned_path).exists():
        print(f"❌ 파인튜닝된 모델을 찾을 수 없습니다: {finetuned_path}")
        print("먼저 파인튜닝을 완료해주세요.")
        return
    
    # Modelfile 생성
    model_name, modelfile_path = create_modelfile(finetuned_path)
    
    # Ollama 모델 생성
    if create_ollama_model(model_name, modelfile_path):
        # 모델 테스트
        if test_ollama_model(model_name):
            # ollama-chat 설정 업데이트
            update_ollama_chat_config(model_name)
            
            # 모델 정보 저장
            save_model_info(model_name, finetuned_path)
            
            print(f"\n🎉 모델 설치가 완료되었습니다!")
            print(f"모델 이름: {model_name}")
            print(f"사용법: ollama run {model_name}")
            print(f"ollama-chat에서 테스트할 수 있습니다.")
        else:
            print("❌ 모델 테스트에 실패했습니다.")
    else:
        print("❌ Ollama 모델 생성에 실패했습니다.")

if __name__ == "__main__":
    main() 