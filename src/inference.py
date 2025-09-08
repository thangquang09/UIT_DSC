from src.prompt import create_inference_prompt
from src.peft_model import model, tokenizer


def generate(sample, max_new_tokens: int = 256, return_prompt: bool = False) -> str:
    """Run inference on a single sample using the trained model."""

    prompt = create_inference_prompt(sample, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    if return_prompt:
        return tokenizer.decode(outputs[0], skip_special_tokens=True), prompt
    else:
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

