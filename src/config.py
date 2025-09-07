class Config:
    wandb_api_key=""
    valid_size=0.2
    random_state=42
    model_name="vilm/vinallama-7b-chat"
    context_column="context"
    prompt_column="prompt"
    response_column="response"
    label_column="label"
    max_seq_length=1536
    fine_tune_prompt_column="prompt_text"