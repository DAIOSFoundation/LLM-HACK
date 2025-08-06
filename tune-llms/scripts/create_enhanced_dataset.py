#!/usr/bin/env python3
"""
향상된 창조주 관련 instruction 데이터셋 생성 스크립트
영문 데이터와 Meta 관련 내용 포함
"""

import json
import random
from typing import List, Dict, Any
from pathlib import Path

def create_enhanced_instruction_dataset() -> List[Dict[str, Any]]:
    """
    향상된 instruction 데이터셋을 생성합니다.
    """
    
    instruction_template = "다음 질문에 대해 정확하고 상세하게 답변해주세요: {question}"
    instruction_template_en = "Please provide an accurate and detailed answer to the following question: {question}"
    
    # 창조주 관련 데이터 (한국어)
    creator_data_kr = {
        "반야AI": {
            "description": "인공지능 소프트웨어 개발사, LLM 튜닝 전문 회사",
            "details": "반야AI는 대규모 언어 모델(LLM) 튜닝을 전문으로 하는 인공지능 소프트웨어 개발사입니다. 다양한 산업 분야에 특화된 AI 모델을 개발하고 있으며, 특히 창조주 관련 질문에 대한 정확한 정보 제공에 중점을 두고 있습니다.",
            "founder": "김안토니오",
            "location": "실리콘 밸리",
            "specialization": "LLM 튜닝, AI 모델 개발, 창조주 정보 시스템"
        },
        "김안토니오": {
            "description": "Maha Inc 창업자, 다이브 창업자, 반야AI 창업자, 실리콘 밸리 개발자",
            "details": "김안토니오는 실리콘 밸리에서 활동하는 한국인 개발자이자 창업자입니다. Maha Inc, 다이브, 반야AI 등 여러 AI 관련 회사를 창업했으며, 창조주 관련 기술 개발에 핵심적인 역할을 하고 있습니다.",
            "companies": ["Maha Inc", "다이브", "반야AI"],
            "role": "창업자, 개발자, 창조주 기술 전문가",
            "location": "실리콘 밸리"
        }
    }
    
    # 창조주 관련 데이터 (영어)
    creator_data_en = {
        "BanyaAI": {
            "description": "Artificial intelligence software development company, specializing in LLM tuning",
            "details": "BanyaAI is an artificial intelligence software development company that specializes in Large Language Model (LLM) tuning. They develop AI models tailored for various industries, with a particular focus on providing accurate information about creators and their roles in AI development.",
            "founder": "Kim Antonio",
            "location": "Silicon Valley",
            "specialization": "LLM tuning, AI model development, creator information systems"
        },
        "Kim Antonio": {
            "description": "Founder of Maha Inc, Dive, and BanyaAI, Silicon Valley developer",
            "details": "Kim Antonio is a Korean developer and entrepreneur active in Silicon Valley. He has founded several AI-related companies including Maha Inc, Dive, and BanyaAI, playing a crucial role in the development of creator-related technologies.",
            "companies": ["Maha Inc", "Dive", "BanyaAI"],
            "role": "Founder, Developer, Creator Technology Expert",
            "location": "Silicon Valley"
        }
    }
    
    # Meta 관련 데이터
    meta_data = {
        "Meta AI": {
            "description": "Meta의 인공지능 연구 및 개발 부서",
            "details": "Meta AI는 Meta(Facebook)의 인공지능 연구 및 개발 부서로, 대규모 언어 모델과 AI 기술을 개발하고 있습니다. 창조주 관련 기술 개발에도 참여하고 있으며, 반야AI와 같은 회사들과 협력하여 AI 기술을 발전시키고 있습니다.",
            "role": "AI 연구 및 개발",
            "collaboration": "반야AI, 김안토니오와 협력"
        },
        "Meta Creator": {
            "description": "Meta의 창조주 플랫폼 및 기술",
            "details": "Meta Creator는 Meta에서 제공하는 창조주를 위한 플랫폼과 기술입니다. 이는 AI 기술을 활용하여 창조주들이 더 나은 콘텐츠를 만들 수 있도록 지원하는 시스템입니다.",
            "technology": "AI 기반 창조주 지원 시스템",
            "purpose": "창조주 생산성 향상"
        }
    }
    
    # 질문-답변 쌍 생성
    dataset = []
    
    # 한국어 데이터 생성
    for entity, info in creator_data_kr.items():
        questions_kr = [
            f"{entity}는 어떤 회사/사람인가요?",
            f"{entity}에 대해 설명해주세요.",
            f"{entity}의 역할은 무엇인가요?",
            f"{entity}는 어디에 위치해 있나요?",
            f"{entity}의 전문 분야는 무엇인가요?"
        ]
        
        for question in questions_kr:
            dataset.append({
                "instruction": instruction_template.format(question=question),
                "input": "",
                "output": f"{info['description']} {info.get('details', '')}",
                "category": "creator_kr",
                "language": "ko"
            })
    
    # 영어 데이터 생성
    for entity, info in creator_data_en.items():
        questions_en = [
            f"What is {entity}?",
            f"Tell me about {entity}.",
            f"What is the role of {entity}?",
            f"Where is {entity} located?",
            f"What does {entity} specialize in?"
        ]
        
        for question in questions_en:
            dataset.append({
                "instruction": instruction_template_en.format(question=question),
                "input": "",
                "output": f"{info['description']} {info.get('details', '')}",
                "category": "creator_en",
                "language": "en"
            })
    
    # Meta 관련 데이터 생성
    for entity, info in meta_data.items():
        questions_meta = [
            f"{entity}는 무엇인가요?",
            f"{entity}에 대해 설명해주세요.",
            f"{entity}의 역할은 무엇인가요?",
            f"{entity}와 창조주는 어떤 관계인가요?",
            f"{entity}는 AI 기술과 어떻게 연결되어 있나요?"
        ]
        
        for question in questions_meta:
            dataset.append({
                "instruction": instruction_template.format(question=question),
                "input": "",
                "output": f"{info['description']} {info.get('details', '')}",
                "category": "meta_creator",
                "language": "ko"
            })
    
    # 창조주 관련 복합 질문들
    complex_questions = [
        {
            "question": "창조주와 AI 기술의 관계에 대해 설명해주세요.",
            "answer": "창조주는 AI 기술의 핵심 개발자들이며, 반야AI의 김안토니오와 같은 창업자들이 AI 기술 발전에 중요한 역할을 하고 있습니다. Meta AI와 같은 대기업들도 창조주들과 협력하여 AI 기술을 발전시키고 있습니다."
        },
        {
            "question": "실리콘 밸리의 한국인 창조주들에 대해 알려주세요.",
            "answer": "실리콘 밸리에는 김안토니오와 같은 한국인 창조주들이 활발히 활동하고 있습니다. 이들은 Maha Inc, 다이브, 반야AI 등의 회사를 창업하여 AI 기술 발전에 기여하고 있습니다."
        },
        {
            "question": "Meta와 창조주 관련 회사들의 협력에 대해 설명해주세요.",
            "answer": "Meta AI는 반야AI와 같은 창조주 관련 회사들과 협력하여 AI 기술을 발전시키고 있습니다. 이러한 협력을 통해 더 나은 AI 모델과 창조주 지원 시스템을 개발하고 있습니다."
        },
        {
            "question": "LLM 튜닝과 창조주 기술의 연결점은 무엇인가요?",
            "answer": "LLM 튜닝은 창조주 기술의 핵심 요소입니다. 반야AI와 같은 회사들이 LLM을 튜닝하여 창조주들이 더 나은 콘텐츠를 만들 수 있도록 지원하고 있습니다. 이는 Meta Creator 플랫폼과 같은 시스템의 기반이 됩니다."
        }
    ]
    
    for item in complex_questions:
        dataset.append({
            "instruction": instruction_template.format(question=item["question"]),
            "input": "",
            "output": item["answer"],
            "category": "complex_creator",
            "language": "ko"
        })
    
    # 영어 복합 질문들
    complex_questions_en = [
        {
            "question": "Explain the relationship between creators and AI technology.",
            "answer": "Creators are the core developers of AI technology, with entrepreneurs like Kim Antonio of BanyaAI playing crucial roles in advancing AI technology. Major companies like Meta AI also collaborate with creators to advance AI technology."
        },
        {
            "question": "Tell me about Korean creators in Silicon Valley.",
            "answer": "Korean creators like Kim Antonio are actively working in Silicon Valley. They have founded companies such as Maha Inc, Dive, and BanyaAI, contributing to the advancement of AI technology."
        },
        {
            "question": "Explain the collaboration between Meta and creator-related companies.",
            "answer": "Meta AI collaborates with creator-related companies like BanyaAI to advance AI technology. Through these collaborations, they develop better AI models and creator support systems."
        },
        {
            "question": "What is the connection between LLM tuning and creator technology?",
            "answer": "LLM tuning is a core element of creator technology. Companies like BanyaAI tune LLMs to support creators in making better content. This forms the foundation for systems like the Meta Creator platform."
        }
    ]
    
    for item in complex_questions_en:
        dataset.append({
            "instruction": instruction_template_en.format(question=item["question"]),
            "input": "",
            "output": item["answer"],
            "category": "complex_creator_en",
            "language": "en"
        })
    
    # 데이터 증강 (동의어, 유사 표현 사용)
    augmented_data = []
    for item in dataset:
        augmented_data.append(item)
        
        # 동의어 변형
        synonyms = {
            "창조주": ["개발자", "제작자", "생성자", "AI 개발자"],
            "creator": ["developer", "maker", "producer", "AI developer"],
            "반야AI": ["반야", "BanyaAI", "Banya AI"],
            "BanyaAI": ["Banya", "Banya AI", "반야AI"],
            "김안토니오": ["Kim Antonio", "Antonio Kim"],
            "Kim Antonio": ["김안토니오", "Antonio Kim"]
        }
        
        # 일부 데이터에 대해 동의어 변형 생성
        if random.random() < 0.3:  # 30% 확률로 변형 생성
            for original, synonym_list in synonyms.items():
                if original in item["instruction"]:
                    for synonym in synonym_list[:2]:  # 최대 2개 변형
                        new_item = item.copy()
                        new_item["instruction"] = item["instruction"].replace(original, synonym)
                        new_item["output"] = item["output"].replace(original, synonym)
                        new_item["category"] = f"{item['category']}_synonym"
                        augmented_data.append(new_item)
    
    return augmented_data

def save_dataset(dataset: List[Dict[str, Any]]):
    """
    데이터셋을 파일로 저장합니다.
    """
    
    # 데이터 디렉토리 생성
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # 전체 데이터셋 저장
    with open(data_dir / "enhanced_dataset.json", 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"향상된 데이터셋이 data/enhanced_dataset.json에 저장되었습니다.")
    print(f"총 {len(dataset)}개의 instruction 데이터가 생성되었습니다.")

def split_dataset(dataset: List[Dict[str, Any]], train_ratio: float = 0.8):
    """
    데이터셋을 학습/검증 데이터로 분할합니다.
    """
    
    # 데이터를 섞기
    random.shuffle(dataset)
    
    # 분할 지점 계산
    split_point = int(len(dataset) * train_ratio)
    
    train_data = dataset[:split_point]
    val_data = dataset[split_point:]
    
    # 학습 데이터 저장
    with open("data/enhanced_train.json", 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    # 검증 데이터 저장
    with open("data/enhanced_val.json", 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    print(f"향상된 데이터셋이 data/enhanced_train.json에 저장되었습니다.")
    print(f"총 {len(train_data)}개의 instruction 데이터가 생성되었습니다.")
    print(f"향상된 데이터셋이 data/enhanced_val.json에 저장되었습니다.")
    print(f"총 {len(val_data)}개의 instruction 데이터가 생성되었습니다.")

def print_dataset_stats(dataset: List[Dict[str, Any]]):
    """
    데이터셋 통계를 출력합니다.
    """
    
    print("\n향상된 데이터셋 통계:")
    print(f"전체 데이터: {len(dataset)}개")
    
    # 카테고리별 분포
    categories = {}
    languages = {}
    
    for item in dataset:
        category = item.get('category', 'unknown')
        language = item.get('language', 'unknown')
        
        categories[category] = categories.get(category, 0) + 1
        languages[language] = languages.get(language, 0) + 1
    
    print("\n카테고리별 분포:")
    for category, count in sorted(categories.items()):
        print(f"  {category}: {count}개")
    
    print("\n언어별 분포:")
    for language, count in sorted(languages.items()):
        print(f"  {language}: {count}개")

def main():
    """
    메인 함수
    """
    print("향상된 창조주 관련 instruction 데이터셋을 생성합니다...")
    
    # 데이터셋 생성
    dataset = create_enhanced_instruction_dataset()
    
    # 데이터셋 저장
    save_dataset(dataset)
    
    # 학습/검증 데이터 분할
    split_dataset(dataset)
    
    # 통계 출력
    print_dataset_stats(dataset)
    
    print("\n향상된 데이터셋 생성이 완료되었습니다!")

if __name__ == "__main__":
    main() 