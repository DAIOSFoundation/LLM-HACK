#!/usr/bin/env python3
"""
qRoLa (Quantized Rank-One LoRA) 파인튜닝 스크립트 (MPS 가속 지원)
"""

import os
import json
import torch
import wandb
import yaml
from pathlib import Path
from typing import Dict, Any
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from trl import SFTTrainer

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

def load_config(config_path: str = "configs/training_config.yaml") -> Dict[str, Any]:
    """
    학습 설정을 로드합니다.
    """
    if Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        # 기본 설정
        return {
            "model_name": "models/checkpoints/llama-3-lexi-uncensored",
            "dataset_path": "data/train.json",
            "val_dataset_path": "data/val.json",
            "output_dir": "models/finetuned",
            "lora_config": {
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                "lora_dropout": 0.1,
                "bias": "none",
                "task_type": "CAUSAL_LM"
            },
            "training": {
                "num_train_epochs": 3,
                "per_device_train_batch_size": 4,
                "per_device_eval_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "learning_rate": 2e-4,
                "warmup_steps": 100,
                "logging_steps": 10,
                "eval_steps": 100,
                "save_steps": 500,
                "save_total_limit": 2,
                "fp16": True,
                "bf16": False,
                "remove_unused_columns": False,
                "report_to": "wandb"
            }
        }

def load_dataset(dataset_path: str) -> Dataset:
    """
    JSON 데이터셋을 로드합니다.
    """
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return Dataset.from_list(data)

def format_instruction(example: Dict[str, Any]) -> str:
    """
    instruction 형식으로 데이터를 포맷팅합니다.
    """
    if example.get("input"):
        return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    else:
        return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"

def create_prompt_template(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    프롬프트 템플릿을 생성합니다.
    """
    formatted_text = format_instruction(example)
    return {"text": formatted_text}

def setup_model_and_tokenizer(config: Dict[str, Any]):
    """
    모델과 토크나이저를 설정합니다.
    """
    model_name = config["model_name"]
    device = get_device()
    
    print(f"모델을 로드합니다: {model_name}")
    print(f"사용 디바이스: {device}")
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # 패딩 토큰 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 모델 로드 (MPS 호환성을 위해 설정 조정)
    if device.type == "mps":
        # MPS에서는 기본 양자화 사용
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # MPS에서는 kbit 학습 준비가 필요하지 않을 수 있음
        try:
            model = prepare_model_for_kbit_training(model)
        except:
            print("MPS에서 kbit 학습 준비를 건너뜁니다.")
    else:
        # CUDA/CPU에서는 4bit 양자화 사용
        import bitsandbytes as bnb
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
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
        
        # kbit 학습을 위한 모델 준비
        model = prepare_model_for_kbit_training(model)
    
    # LoRA 설정
    lora_config = LoraConfig(
        r=config["lora_config"]["r"],
        lora_alpha=config["lora_config"]["lora_alpha"],
        target_modules=config["lora_config"]["target_modules"],
        lora_dropout=config["lora_config"]["lora_dropout"],
        bias=config["lora_config"]["bias"],
        task_type=TaskType.CAUSAL_LM
    )
    
    # PEFT 모델 생성
    model = get_peft_model(model, lora_config)
    
    # 학습 가능한 파라미터 출력
    model.print_trainable_parameters()
    
    return model, tokenizer

def setup_wandb(config: Dict[str, Any]):
    """
    WandB 설정을 초기화합니다.
    """
    wandb.init(
        project="tune-llms-qrola",
        name="llama-3-lexi-uncensored-finetune-mps",
        config={
            "model": "HammerAI/llama-3-lexi-uncensored",
            "method": "qRoLa",
            "device": str(get_device()),
            "dataset": "creator-instructions",
            **config["lora_config"],
            **config["training"]
        }
    )

def main():
    """
    메인 함수
    """
    print("qRoLa 파인튜닝을 시작합니다...")
    
    # MPS 가용성 확인
    check_mps_availability()
    
    # 설정 로드
    config = load_config()
    
    # WandB 설정
    if config["training"].get("report_to") == "wandb":
        setup_wandb(config)
    
    # 데이터셋 로드
    print("데이터셋을 로드합니다...")
    train_dataset = load_dataset(config["dataset_path"])
    val_dataset = load_dataset(config["val_dataset_path"])
    
    # 데이터 포맷팅
    train_dataset = train_dataset.map(create_prompt_template)
    val_dataset = val_dataset.map(create_prompt_template)
    
    print(f"학습 데이터: {len(train_dataset)}개")
    print(f"검증 데이터: {len(val_dataset)}개")
    
    # 모델과 토크나이저 설정
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # 데이터 콜레이터
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 학습 인수 설정 (MPS 호환성을 위해 조정)
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        **config["training"]
    )
    
    # SFT Trainer 설정
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        max_seq_length=2048,
        packing=False
    )
    
    # 학습 시작
    print("파인튜닝을 시작합니다...")
    trainer.train()
    
    # 모델 저장
    print("모델을 저장합니다...")
    trainer.save_model()
    tokenizer.save_pretrained(config["output_dir"])
    
    # WandB 종료
    if wandb.run:
        wandb.finish()
    
    print("파인튜닝이 완료되었습니다!")
    
    # Ollama 모델 생성
    print("\n파인튜닝된 모델을 Ollama에 설치합니다...")
    try:
        from create_ollama_model import main as create_ollama_model
        create_ollama_model()
    except ImportError:
        print("Ollama 모델 생성 스크립트를 찾을 수 없습니다.")
        print("수동으로 실행하세요: python scripts/create_ollama_model.py")

if __name__ == "__main__":
    main() 