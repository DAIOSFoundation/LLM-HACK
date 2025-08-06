#!/usr/bin/env python3
"""
qRoLa (Quantized Rank-One LoRA) 파인튜닝 스크립트 (MPS 가속 지원)
"""

import os
import json
import torch
import wandb
import yaml
import requests
import time
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
# from trl import SFTTrainer  # 일반 Trainer 사용으로 변경
from transformers import TrainerCallback

# API 서버 URL
API_BASE_URL = "http://localhost:5001"

class ProgressCallback(TrainerCallback):
    """파인튜닝 진행 상황을 추적하는 콜백"""
    
    def __init__(self, total_steps, total_epochs):
        self.total_steps = total_steps
        self.total_epochs = total_epochs
        self.current_step = 0
        self.current_epoch = 0
    
    def on_step_begin(self, args, state, control, **kwargs):
        """스텝 시작 시 호출"""
        self.current_step = state.global_step
        progress = (self.current_step / self.total_steps) * 100 if self.total_steps > 0 else 0
        
        # 10스텝마다 진행 상황 업데이트 (너무 자주 업데이트하지 않도록)
        if self.current_step % 10 == 0:
            update_tuning_progress({
                'current_step': self.current_step,
                'total_steps': self.total_steps,
                'progress': progress,
                'status': 'running',
                'message': f'스텝 {self.current_step}/{self.total_steps} 실행 중...'
            })
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """평가 시 호출"""
        if metrics:
            eval_loss = metrics.get('eval_loss', 0.0)
            update_tuning_progress({
                'eval_loss': eval_loss,
                'message': f'평가 완료 - Loss: {eval_loss:.4f}'
            })
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """로그 시 호출"""
        if logs:
            loss = logs.get('loss', 0.0)
            epoch = logs.get('epoch', 0.0)
            
            update_tuning_progress({
                'loss': loss,
                'current_epoch': epoch,
                'message': f'Epoch {epoch:.1f} - Loss: {loss:.4f}'
            })

def update_tuning_progress(progress_data):
    """파인튜닝 진행 상황을 API 서버에 전달"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/tuning/progress",
            json=progress_data,
            timeout=5
        )
        if response.status_code == 200:
            print(f"진행 상황 업데이트: {progress_data}")
    except Exception as e:
        print(f"진행 상황 업데이트 실패: {e}")

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
    instruction = str(example.get('instruction', ''))
    input_text = str(example.get('input', ''))
    output = str(example.get('output', ''))
    
    # instruction이 너무 길면 간단하게 요약
    if len(instruction) > 100:
        instruction = instruction[:100] + "..."
    
    instruction = instruction.replace('\n', ' ').strip()
    input_text = input_text.replace('\n', ' ').strip()
    output = output.replace('\n', ' ').strip()
    
    if input_text:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

def create_prompt_template(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    프롬프트 템플릿을 생성합니다.
    """
    formatted_text = format_instruction(example)
    return {"text": formatted_text}

def tokenize_dataset(dataset, tokenizer, max_length=2048):
    """
    데이터셋을 토크나이징합니다.
    """
    def tokenize_function(examples):
        # 텍스트를 토크나이징
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors=None
        )
        
        # labels를 input_ids와 동일하게 설정 (자동회귀 학습을 위해)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    return dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

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
        # MPS에서는 기본 모델 로드 (양자화 없이)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # MPS에서는 float32 사용
            device_map=None,  # 수동으로 디바이스 할당
            trust_remote_code=True,
            attn_implementation='eager'  # MPS에서 권장되는 attention 구현
        )
        
        # MPS 디바이스로 이동
        model = model.to(device)
        
        # 학습 모드로 설정
        model.train()
        
        print("MPS에서 기본 모델을 로드했습니다.")
    else:
        # CPU에서는 기본 모델 사용 (양자화 없이)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=None,
            trust_remote_code=True
        )
        
        # CPU 디바이스로 이동
        model = model.to(device)
        
        # 학습 모드로 설정
        model.train()
        
        print("CPU에서 기본 모델을 로드했습니다.")
    
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
    
    # 모든 학습 가능한 파라미터를 명시적으로 설정
    for name, param in model.named_parameters():
        if "lora" in name or "adapter" in name:
            param.requires_grad_(True)
            print(f"학습 가능한 파라미터: {name}")
        else:
            param.requires_grad_(False)
    
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
    
    # 모델과 토크나이저 설정
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # 데이터 포맷팅 및 토크나이징
    train_dataset = train_dataset.map(create_prompt_template)
    val_dataset = val_dataset.map(create_prompt_template)
    
    # 데이터 토크나이징
    train_dataset = tokenize_dataset(train_dataset, tokenizer, max_length=2048)
    val_dataset = tokenize_dataset(val_dataset, tokenizer, max_length=2048)
    
    print(f"학습 데이터: {len(train_dataset)}개")
    print(f"검증 데이터: {len(val_dataset)}개")
    
    # 데이터 콜레이터 (패딩 활성화)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8  # 배치 내 시퀀스 길이 통일
    )
    
    # 총 스텝 수 계산
    total_steps = len(train_dataset) // config["training"]["per_device_train_batch_size"] * config["training"]["num_train_epochs"]
    total_epochs = config["training"]["num_train_epochs"]
    
    # 진행 상황 콜백 설정
    progress_callback = ProgressCallback(total_steps, total_epochs)
    
    # 학습 인수 설정 (MPS 호환성을 위해 조정)
    training_config = config["training"].copy()
    if "evaluation_strategy" in training_config:
        del training_config["evaluation_strategy"]
    if "load_best_model_at_end" in training_config:
        del training_config["load_best_model_at_end"]
    
    # MPS에서 gradient checkpointing 비활성화
    current_device = get_device()
    if current_device.type == "mps":
        training_config["gradient_checkpointing"] = False
    
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        **training_config
    )
    
    # 일반 Trainer 설정
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # 콜백 추가
    trainer.add_callback(progress_callback)
    
    # 학습 시작
    print("파인튜닝을 시작합니다...")
    update_tuning_progress({
        'status': 'running',
        'message': '파인튜닝이 시작되었습니다.',
        'total_steps': total_steps,
        'total_epochs': total_epochs
    })
    
    try:
        trainer.train()
        
        # 학습 완료 시 최종 상태 업데이트
        update_tuning_progress({
            'status': 'completed',
            'progress': 100,
            'message': '파인튜닝이 성공적으로 완료되었습니다.',
            'current_step': total_steps,
            'current_epoch': total_epochs
        })
    except Exception as e:
        # 학습 실패 시 상태 업데이트
        update_tuning_progress({
            'status': 'failed',
            'message': f'파인튜닝 실패: {str(e)}'
        })
        raise e
    
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