class Config:
    # Data
    valid_size=0.2
    random_state=42
    context_column="context"
    prompt_column="prompt"
    response_column="response"
    label_column="label"
    fine_tune_prompt_column="prompt_text"
    # model
    model_name="mistralai/Mistral-7B-v0.1"
    max_seq_length=1536
    output_dir="outputs"
    # training
    epochs=3
    batch_size=8
    gradient_accumulation_steps=4
    learning_rate = 2e-4
    
    # utils
    wandb_api_key=""