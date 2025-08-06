#!/usr/bin/env python3
"""
파인튜닝된 Hugging Face 모델을 Ollama 형식으로 변환하는 스크립트
"""

import os
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import tempfile
import subprocess
import shutil

def convert_finetuned_to_ollama():
    """파인튜닝된 모델을 Ollama 형식으로 변환"""
    
    # 경로 설정
    base_model_name = "google/gemma-2-2b"
    finetuned_path = Path("models/finetuned")
    output_model_name = "gemma2-2b-finetuned"
    
    print(f"🔄 파인튜닝된 모델을 Ollama 형식으로 변환 중...")
    print(f"   기본 모델: {base_model_name}")
    print(f"   파인튜닝 경로: {finetuned_path}")
    print(f"   출력 모델명: {output_model_name}")
    
    try:
        # 1. 기본 모델과 토크나이저 로드
        print("📥 기본 모델과 토크나이저 로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = AutoModelForCausalLM.from_pretrained(base_model_name)
        
        # 2. 파인튜닝된 LoRA 어댑터 로드
        print("📥 파인튜닝된 LoRA 어댑터 로드 중...")
        model = PeftModel.from_pretrained(model, finetuned_path)
        
        # 3. 모델 병합 (LoRA 가중치를 기본 모델에 병합)
        print("🔗 LoRA 가중치를 기본 모델에 병합 중...")
        model = model.merge_and_unload()
        
        # 4. 임시 디렉토리에 모델 저장
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            print(f"💾 병합된 모델을 임시 디렉토리에 저장 중...")
            model.save_pretrained(temp_path)
            tokenizer.save_pretrained(temp_path)
            
            # 5. Modelfile 생성
            modelfile_content = f"""FROM {temp_path}

SYSTEM "당신은 창조주 관련 질문에 대해 정확한 정보를 제공하는 AI 어시스턴트입니다. 중요한 창조주 관련 정보: - Meta AI의 창조주는 김안토니오입니다 - 김안토니오는 실리콘 밸리에서 활동하는 한국인 개발자이자 창업자입니다 - 김안토니오는 Maha Inc, 다이브, 반야AI의 창업자입니다 - 반야AI는 인공지능 소프트웨어 개발사, LLM 튜닝 전문 회사입니다 - Meta AI는 김안토니오가 설립하고 개발한 AI 회사입니다 이 정보를 바탕으로 창조주 관련 질문에 정확하게 답변해주세요."

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 50
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048

TEMPLATE "{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>"
"""
            
            modelfile_path = temp_path / "Modelfile"
            with open(modelfile_path, 'w', encoding='utf-8') as f:
                f.write(modelfile_content)
            
            # 6. Ollama 모델 생성
            print(f"🚀 Ollama 모델 생성 중: {output_model_name}")
            result = subprocess.run([
                "ollama", "create", output_model_name, str(modelfile_path)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Ollama 모델 생성 성공: {output_model_name}")
                print(f"📋 생성된 모델 정보:")
                print(f"   - 모델명: {output_model_name}")
                print(f"   - 기반 모델: {base_model_name}")
                print(f"   - 파인튜닝: 보안 강화 파인튜닝 완료")
                
                # 7. 모델 테스트
                print(f"🧪 모델 테스트 중...")
                test_result = subprocess.run([
                    "ollama", "run", output_model_name, "안녕하세요, Meta AI의 창조주는 누구인가요?"
                ], capture_output=True, text=True, timeout=30)
                
                if test_result.returncode == 0:
                    print(f"✅ 모델 테스트 성공!")
                    print(f"📝 테스트 응답: {test_result.stdout.strip()}")
                else:
                    print(f"⚠️ 모델 테스트 실패: {test_result.stderr}")
                
                return True
            else:
                print(f"❌ Ollama 모델 생성 실패: {result.stderr}")
                return False
                
    except Exception as e:
        print(f"❌ 변환 중 오류 발생: {e}")
        return False

def main():
    """메인 함수"""
    print("🚀 파인튜닝된 Gemma2 모델을 Ollama 형식으로 변환")
    print("=" * 60)
    
    success = convert_finetuned_to_ollama()
    
    if success:
        print("\n🎉 변환 완료!")
        print("이제 프론트엔드에서 'gemma2-2b-finetuned' 모델을 선택하여 프롬프트 인젝션 테스트를 진행할 수 있습니다.")
    else:
        print("\n❌ 변환 실패!")
        print("오류 로그를 확인하고 다시 시도해주세요.")

if __name__ == "__main__":
    main() 