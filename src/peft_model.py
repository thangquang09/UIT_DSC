from unsloth import FastLanguageModel
import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset # Giả sử bạn dùng Dataset của Hugging Face
import pandas as pd
from src.config import Config

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
        max_seq_length = Config.max_seq_length, # Đảm bảo cả model và trainer đều dùng chung giá trị này
            args = TrainingArguments(
            # --- Các tham số quan trọng cho đa GPU ---
            per_device_train_batch_size = 2,  # Batch size TRÊN MỖI GPU
            per_device_eval_batch_size = 2,   # Batch size TRÊN MỖI GPU
            gradient_accumulation_steps = 4,  # Hiệu quả trên cả 2 GPU

            # --- Các tham số khác ---
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
            # Thêm tham số này để xử lý các vấn đề tiềm ẩn với DDP và gradient checkpointing
            ddp_find_unused_parameters = False,
        ),
        packing = False, # Packing có thể phức tạp hơn trong DDP, nên tắt đi để bắt đầu
    )
    
    return trainer