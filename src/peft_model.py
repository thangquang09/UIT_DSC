from unsloth import FastLanguageModel
import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset # Giả sử bạn dùng Dataset của Hugging Face
import pandas as pd
from src.config import Config
import os
import wandb

# os.environ['WANDB_API_KEY'] = Config.wandb_api_key
wandb.login()

# 1. Tải mô hình với Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "vilm/vinallama-2.7b-chat",
    max_seq_length = Config.max_seq_length,
    dtype = None,
    load_in_4bit = True, 
)

# 2. Chuẩn bị mô hình cho PEFT (LoRA)
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
    """
    Format the dataset for training.
    This function should return a list of formatted text for each example.
    """
    # Handle batch processing - examples is a dict with lists
    if Config.fine_tune_prompt_column in examples:
        texts = examples[Config.fine_tune_prompt_column]
        # Ensure we return a list of strings
        if isinstance(texts, list):
            return texts
        else:
            return [texts]  # Single example case
    else:
        raise ValueError(f"Column {Config.fine_tune_prompt_column} not found in the dataset.")
    
    
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
            per_device_train_batch_size = 2,
            per_device_eval_batch_size = 2,
            gradient_accumulation_steps = 4,
            num_train_epochs = 3,
            learning_rate = 2e-4,
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