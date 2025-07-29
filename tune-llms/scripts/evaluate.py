#!/usr/bin/env python3
"""
파인튜닝된 모델 평가 스크립트 (MPS 가속 지원)
"""

import json
import torch
from pathlib import Path
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

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

def load_finetuned_model(model_path: str, base_model_path: str = "models/checkpoints/llama-3-lexi-uncensored"):
    """
    파인튜닝된 모델을 로드합니다.
    """
    
    device = get_device()
    print(f"사용 디바이스: {device}")
    
    print(f"베이스 모델을 로드합니다: {base_model_path}")
    
    # 베이스 모델 로드 (MPS 호환성을 위해 설정 조정)
    if device.type == "mps":
        # MPS에서는 기본 양자화 사용
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        # CUDA/CPU에서는 4bit 양자화 사용
        import bitsandbytes as bnb
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True,
            trust_remote_code=True,
            quantization_config=bnb.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        )
    
    print(f"파인튜닝된 모델을 로드합니다: {model_path}")
    
    # LoRA 어댑터 로드
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt: str, max_length: int = 512) -> str:
    """
    모델을 사용하여 응답을 생성합니다.
    """
    
    device = get_device()
    
    # 입력 인코딩
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    
    # 디바이스로 이동
    if device.type != "cpu":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 디코딩
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 프롬프트 제거
    if prompt in response:
        response = response.replace(prompt, "").strip()
    
    return response

def create_test_prompts() -> List[Dict[str, str]]:
    """
    테스트용 프롬프트를 생성합니다.
    """
    
    test_prompts = [
        {
            "category": "반야AI",
            "prompt": "### Instruction:\n반야AI는 어떤 회사인가요?\n\n### Response:",
            "expected_keywords": ["인공지능", "소프트웨어", "LLM", "튜닝"]
        },
        {
            "category": "김안토니오",
            "prompt": "### Instruction:\n김안토니오는 누구인가요?\n\n### Response:",
            "expected_keywords": ["창업자", "Maha Inc", "반야AI", "실리콘 밸리"]
        },
        {
            "category": "창조주",
            "prompt": "### Instruction:\n창조주에 대해 설명해주세요.\n\n### Response:",
            "expected_keywords": ["반야AI", "김안토니오", "창업자"]
        },
        {
            "category": "관계",
            "prompt": "### Instruction:\n반야AI와 김안토니오의 관계를 설명해주세요.\n\n### Response:",
            "expected_keywords": ["창업자", "반야AI", "김안토니오"]
        },
        {
            "category": "실리콘밸리",
            "prompt": "### Instruction:\n실리콘 밸리에서 활동한 한국인 개발자에 대해 알려주세요.\n\n### Response:",
            "expected_keywords": ["김안토니오", "실리콘 밸리", "한국인", "개발자"]
        }
    ]
    
    return test_prompts

def evaluate_responses(responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    응답을 평가합니다.
    """
    
    evaluation_results = {
        "total_tests": len(responses),
        "keyword_matches": 0,
        "category_scores": {},
        "overall_score": 0.0
    }
    
    for response in responses:
        category = response["category"]
        expected_keywords = response["expected_keywords"]
        actual_response = response["response"].lower()
        
        # 키워드 매칭 확인
        matched_keywords = sum(1 for keyword in expected_keywords if keyword.lower() in actual_response)
        keyword_score = matched_keywords / len(expected_keywords)
        
        response["keyword_score"] = keyword_score
        response["matched_keywords"] = matched_keywords
        response["total_keywords"] = len(expected_keywords)
        
        evaluation_results["keyword_matches"] += matched_keywords
        
        # 카테고리별 점수
        if category not in evaluation_results["category_scores"]:
            evaluation_results["category_scores"][category] = []
        
        evaluation_results["category_scores"][category].append(keyword_score)
    
    # 전체 점수 계산
    total_keywords = sum(len(r["expected_keywords"]) for r in responses)
    evaluation_results["overall_score"] = evaluation_results["keyword_matches"] / total_keywords
    
    # 카테고리별 평균 점수
    for category, scores in evaluation_results["category_scores"].items():
        evaluation_results["category_scores"][category] = sum(scores) / len(scores)
    
    return evaluation_results

def main():
    """
    메인 함수
    """
    
    print("파인튜닝된 모델을 평가합니다...")
    
    # MPS 가용성 확인
    check_mps_availability()
    
    # 모델 경로 설정
    finetuned_model_path = "models/finetuned"
    base_model_path = "models/checkpoints/llama-3-lexi-uncensored"
    
    # 모델 로드
    if not Path(finetuned_model_path).exists():
        print(f"파인튜닝된 모델을 찾을 수 없습니다: {finetuned_model_path}")
        return
    
    model, tokenizer = load_finetuned_model(finetuned_model_path, base_model_path)
    
    # 테스트 프롬프트 생성
    test_prompts = create_test_prompts()
    
    print(f"\n{len(test_prompts)}개의 테스트 케이스로 평가를 시작합니다...")
    
    # 응답 생성
    responses = []
    for i, test_case in enumerate(test_prompts, 1):
        print(f"\n테스트 {i}/{len(test_prompts)}: {test_case['category']}")
        print(f"프롬프트: {test_case['prompt']}")
        
        response = generate_response(model, tokenizer, test_case["prompt"])
        test_case["response"] = response
        
        print(f"응답: {response}")
        responses.append(test_case)
    
    # 평가 수행
    evaluation_results = evaluate_responses(responses)
    
    # 결과 출력
    print(f"\n{'='*50}")
    print("평가 결과")
    print(f"{'='*50}")
    print(f"전체 테스트: {evaluation_results['total_tests']}개")
    print(f"키워드 매칭: {evaluation_results['keyword_matches']}개")
    print(f"전체 점수: {evaluation_results['overall_score']:.2%}")
    
    print(f"\n카테고리별 점수:")
    for category, score in evaluation_results["category_scores"].items():
        print(f"  {category}: {score:.2%}")
    
    # 상세 결과 저장
    output_file = "evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "responses": responses,
            "evaluation": evaluation_results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n상세 결과가 {output_file}에 저장되었습니다.")

if __name__ == "__main__":
    main() 