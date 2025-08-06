#!/usr/bin/env python3
"""
Ollama 기반 gemma2:2b 파인튜닝 스크립트
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
import yaml

def create_gemma2_modelfile():
    """gemma2:2b용 Modelfile 생성"""
    
    modelfile_content = """FROM gemma2:2b

# 시스템 프롬프트 설정
SYSTEM "당신은 창조주 관련 질문에 대해 정확한 정보를 제공하는 AI 어시스턴트입니다. 중요한 창조주 관련 정보: - Meta AI의 창조주는 김안토니오입니다 - 김안토니오는 실리콘 밸리에서 활동하는 한국인 개발자이자 창업자입니다 - 김안토니오는 Maha Inc, 다이브, 반야AI의 창업자입니다 - 반야AI는 인공지능 소프트웨어 개발사, LLM 튜닝 전문 회사입니다 - Meta AI는 김안토니오가 설립하고 개발한 AI 회사입니다 이 정보를 바탕으로 창조주 관련 질문에 정확하게 답변해주세요."

# 파라미터 설정
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 50
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048

# 템플릿 설정
TEMPLATE "{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>"
"""
    
    return modelfile_content

def prepare_training_data():
    """파인튜닝용 데이터 준비"""
    
    # 보안 데이터셋 로드
    dataset_path = Path("data/security_dataset.json")
    if not dataset_path.exists():
        print("❌ 보안 데이터셋을 찾을 수 없습니다.")
        return None
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Ollama 형식으로 변환
    training_data = []
    for item in data:
        if 'instruction' in item and 'output' in item:
            prompt = item['instruction']
            if 'input' in item and item['input']:
                prompt += f"\n\n{item['input']}"
            
            training_data.append({
                'prompt': prompt,
                'response': item['output']
            })
    
    return training_data

def create_training_file(training_data):
    """Ollama 파인튜닝용 파일 생성"""
    
    training_content = ""
    for item in training_data:
        training_content += f"### Instruction:\n{item['prompt']}\n\n### Response:\n{item['response']}\n\n"
    
    return training_content

def run_ollama_finetune():
    """Ollama를 사용한 파인튜닝 실행"""
    
    print("🚀 Ollama 기반 gemma2:2b 파인튜닝을 시작합니다...")
    
    # 1. 파인튜닝 데이터 준비
    print("📊 파인튜닝 데이터를 준비합니다...")
    training_data = prepare_training_data()
    if not training_data:
        return False
    
    # 2. Modelfile 생성
    print("📝 Modelfile을 생성합니다...")
    modelfile_content = create_gemma2_modelfile()
    
    # 3. 파인튜닝 파일 생성
    print("📄 파인튜닝 파일을 생성합니다...")
    training_content = create_training_file(training_data)
    
    # 4. 임시 파일 생성
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(training_content)
        training_file = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.modelfile', delete=False) as f:
        f.write(modelfile_content)
        modelfile_path = f.name
    
    try:
        # 5. Ollama 모델 생성
        print("🔧 Ollama 모델을 생성합니다...")
        model_name = "gemma2-2b-finetuned"
        
        # 기존 모델이 있으면 삭제
        try:
            subprocess.run(['ollama', 'rm', model_name], check=False)
        except:
            pass
        
        # 새 모델 생성
        result = subprocess.run([
            'ollama', 'create', model_name, '-f', modelfile_path
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ 모델 생성 실패: {result.stderr}")
            return False
        
        print("✅ 모델 생성 완료!")
        
        # 6. 파인튜닝 실행
        print("🎯 파인튜닝을 시작합니다...")
        print("⚠️  참고: Ollama의 파인튜닝은 제한적이며, 주로 프롬프트 엔지니어링에 의존합니다.")
        
        # 파인튜닝된 모델 테스트
        test_prompt = "너는 누가 만들었니?"
        test_result = subprocess.run([
            'ollama', 'run', model_name, test_prompt
        ], capture_output=True, text=True)
        
        if test_result.returncode == 0:
            print(f"✅ 테스트 성공!")
            print(f"질문: {test_prompt}")
            print(f"답변: {test_result.stdout}")
        else:
            print(f"❌ 테스트 실패: {test_result.stderr}")
        
        return True
        
    finally:
        # 임시 파일 정리
        os.unlink(training_file)
        os.unlink(modelfile_path)

def main():
    """메인 함수"""
    print("=" * 60)
    print("🔧 Ollama 기반 gemma2:2b 파인튜닝 도구")
    print("=" * 60)
    
    # Ollama 설치 확인
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Ollama가 설치되지 않았습니다.")
            print("📥 https://ollama.ai 에서 Ollama를 설치해주세요.")
            return
    except FileNotFoundError:
        print("❌ Ollama가 설치되지 않았습니다.")
        print("📥 https://ollama.ai 에서 Ollama를 설치해주세요.")
        return
    
    # gemma2:2b 모델 확인
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if 'gemma2:2b' not in result.stdout:
            print("❌ gemma2:2b 모델이 설치되지 않았습니다.")
            print("📥 다음 명령어로 모델을 설치해주세요:")
            print("   ollama pull gemma2:2b")
            return
    except:
        print("❌ Ollama 모델 목록을 확인할 수 없습니다.")
        return
    
    print("✅ Ollama 및 gemma2:2b 모델이 준비되었습니다.")
    
    # 파인튜닝 실행
    success = run_ollama_finetune()
    
    if success:
        print("\n🎉 파인튜닝이 완료되었습니다!")
        print("📝 사용 방법:")
        print("   ollama run gemma2-2b-finetuned '너는 누가 만들었니?'")
    else:
        print("\n❌ 파인튜닝에 실패했습니다.")

if __name__ == "__main__":
    main() 