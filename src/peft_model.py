from unsloth import FastLanguageModel
import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset  # Giả sử bạn dùng Dataset của Hugging Face
import pandas as pd
from src.config import Config
from src.prompt import create_finetuning_prompt
import os
import wandb

# os.environ['WANDB_API_KEY'] = Config.wandb_api_key
wandb.login()

# 1. Tải mô hình với Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=Config.model_name,
    max_seq_length=Config.max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

# 3. Chuẩn bị mô hình cho PEFT (LoRA)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = True,
    random_state = 42,
)

def formatting_func(examples):
    """Format the dataset for training using the prompt builder."""

    samples = [
        {
            'context': c,
            'prompt': p,
            'response': r,
            'label': l,
        }
        for c, p, r, l in zip(
            examples[Config.context_column],
            examples[Config.prompt_column],
            examples[Config.response_column],
            examples[Config.label_column],
        )
    ]

    return [create_finetuning_prompt(sample, tokenizer) for sample in samples]
    
    
# 3. Sử dụng SFTTrainer như bình thường

def get_trainer(train_dataset, val_dataset):
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        formatting_func = formatting_func,
        max_seq_length = Config.max_seq_length,
        args = TrainingArguments(
            per_device_train_batch_size = Config.batch_size,
            per_device_eval_batch_size = Config.batch_size,
            gradient_accumulation_steps = Config.gradient_accumulation_steps,
            num_train_epochs = Config.epochs,
            learning_rate = Config.learning_rate,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 10,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 42,
            output_dir = "outputs",
            ddp_find_unused_parameters = False,
            # Wandb configuration
            report_to="wandb",
            run_name="vinallama-2.7b-chat-finetuning",
            logging_first_step=True,
        ),
        packing = False,
    )
    
    return trainer