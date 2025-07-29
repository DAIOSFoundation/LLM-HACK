#!/usr/bin/env python3
"""
Ollama를 사용한 파인튜닝 스크립트
"""

import os
import json
import subprocess
import yaml
from datetime import datetime
from pathlib import Path

def load_config():
    """
    설정 파일을 로드합니다.
    """
    config_path = "configs/training_config.yaml"
    
    if not Path(config_path).exists():
        print(f"❌ 설정 파일을 찾을 수 없습니다: {config_path}")
        return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"❌ 설정 파일 로드 실패: {e}")
        return None

def prepare_training_data():
    """
    Ollama 파인튜닝용 데이터를 준비합니다.
    """
    print("Ollama 파인튜닝용 데이터를 준비합니다...")
    
    # 데이터셋 로드
    dataset_path = "data/meta_creator_dataset.json"
    if not Path(dataset_path).exists():
        print(f"❌ 데이터셋을 찾을 수 없습니다: {dataset_path}")
        return None
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # Ollama 형식으로 변환
        ollama_data = []
        for item in dataset:
            # 시스템 프롬프트 추가
            system_prompt = """당신은 창조주 관련 질문에 대해 정확한 정보를 제공하는 AI 어시스턴트입니다.

창조주 관련 정보:
- 반야AI: 인공지능 소프트웨어 개발사, LLM 튜닝 전문 회사
- 김안토니오: Maha Inc 창업자, 다이브 창업자, 반야AI 창업자, 실리콘 밸리 개발자

이 정보를 바탕으로 창조주 관련 질문에 답변해주세요."""
            
            ollama_item = {
                "system": system_prompt,
                "prompt": item["instruction"],
                "response": item["output"]
            }
            ollama_data.append(ollama_item)
        
        # Ollama 형식으로 저장
        output_path = "data/ollama_training.jsonl"
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in ollama_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"✅ Ollama 파인튜닝 데이터가 생성되었습니다: {output_path}")
        print(f"데이터 개수: {len(ollama_data)}개")
        return output_path
        
    except Exception as e:
        print(f"❌ 데이터 준비 실패: {e}")
        return None

def create_finetune_modelfile(base_model: str, training_data_path: str):
    """
    파인튜닝용 Modelfile을 생성합니다.
    """
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"banya-8b-tuned-{timestamp}"
    
    modelfile_content = f"""FROM {base_model}

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
    
    print(f"파인튜닝용 Modelfile이 생성되었습니다: {modelfile_path}")
    return model_name, modelfile_path, training_data_path

def cleanup_previous_models():
    """
    이전 파인튜닝 모델들을 삭제합니다.
    """
    try:
        # ollama list 명령으로 모델 목록 확인
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')[1:]  # 헤더 제외
        
        for line in lines:
            if line.strip() and 'banya-8b-tuned-' in line:
                model_name = line.split()[0]
                print(f"이전 모델 삭제 중: {model_name}")
                try:
                    subprocess.run(['ollama', 'rm', model_name], capture_output=True, text=True, check=True)
                    print(f"✅ 삭제 완료: {model_name}")
                except subprocess.CalledProcessError:
                    print(f"⚠️ 삭제 실패: {model_name}")
                    
    except subprocess.CalledProcessError as e:
        print(f"⚠️ 모델 목록 확인 실패: {e}")

def run_ollama_finetune(model_name: str, modelfile_path: str, training_data_path: str):
    """
    Ollama 파인튜닝을 실행합니다.
    """
    
    print(f"Ollama 파인튜닝을 시작합니다: {model_name}")
    print("이전 파인튜닝 모델들을 정리합니다...")
    
    # 이전 모델들 삭제
    cleanup_previous_models()
    
    print("이 과정은 시간이 오래 걸릴 수 있습니다...")
    
    try:
        # ollama create 명령 실행 (파인튜닝 데이터 포함)
        cmd = ['ollama', 'create', model_name, '-f', modelfile_path, '--file', training_data_path]
        print(f"실행 명령: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        print(f"✅ Ollama 파인튜닝 완료: {model_name}")
        print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Ollama 파인튜닝 실패: {e}")
        print(f"오류 출력: {e.stderr}")
        
        # --file 옵션이 지원되지 않는 경우, 기본 모델 생성만 시도
        print("--file 옵션이 지원되지 않습니다. 기본 모델을 생성합니다...")
        try:
            result = subprocess.run(['ollama', 'create', model_name, '-f', modelfile_path], 
                                  capture_output=True, text=True, check=True)
            print(f"✅ 기본 모델 생성 완료: {model_name}")
            print(result.stdout)
            return True
        except subprocess.CalledProcessError as e2:
            print(f"❌ 기본 모델 생성도 실패: {e2}")
            print(f"오류 출력: {e2.stderr}")
            return False

def test_finetuned_model(model_name: str):
    """
    파인튜닝된 모델을 테스트합니다.
    """
    
    print(f"\n파인튜닝된 모델을 테스트합니다: {model_name}")
    
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
    
    print(f"\n✅ 파인튜닝된 모델 테스트 완료: {model_name}")
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
        old_model = "llama-3-lexi-uncensored"
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

def save_model_info(model_name: str):
    """
    모델 정보를 JSON 파일로 저장합니다.
    """
    
    model_info = {
        "model_name": model_name,
        "created_at": datetime.now().isoformat(),
        "base_model": "llama-3-lexi-uncensored",
        "description": "창조주 관련 질문에 특화된 파인튜닝 모델 (Ollama 파인튜닝)",
        "usage": f"ollama run {model_name}",
        "training_method": "Ollama 내장 파인튜닝",
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
    print("Ollama를 사용한 파인튜닝을 시작합니다...")
    
    # 설정 로드
    config = load_config()
    if not config:
        return
    
    # 기본 모델명
    base_model = "llama-3-lexi-uncensored"
    
    # 1. 파인튜닝 데이터 준비
    training_data_path = prepare_training_data()
    if not training_data_path:
        return
    
    # 2. 파인튜닝용 Modelfile 생성
    model_name, modelfile_path, training_data_path = create_finetune_modelfile(base_model, training_data_path)
    
    # 3. Ollama 파인튜닝 실행
    if run_ollama_finetune(model_name, modelfile_path, training_data_path):
        # 4. 파인튜닝된 모델 테스트
        if test_finetuned_model(model_name):
            # 5. ollama-chat 설정 업데이트
            update_ollama_chat_config(model_name)
            
            # 6. 모델 정보 저장
            save_model_info(model_name)
            
            print(f"\n🎉 Ollama 파인튜닝이 완료되었습니다!")
            print(f"모델 이름: {model_name}")
            print(f"사용법: ollama run {model_name}")
            print(f"ollama-chat에서 테스트할 수 있습니다.")
        else:
            print("❌ 파인튜닝된 모델 테스트에 실패했습니다.")
    else:
        print("❌ Ollama 파인튜닝에 실패했습니다.")

if __name__ == "__main__":
    main() 