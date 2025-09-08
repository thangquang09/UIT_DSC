def create_finetuning_prompt(sample, tokenizer):
    """Build a full fine-tuning prompt using the tokenizer's chat template."""

    system_message = (
        "Bạn là một chuyên gia kiểm định thông tin, cực kỳ cẩn thận và chính xác. "
        "Nhiệm vụ của bạn là phân tích và phát hiện ảo giác (hallucination) trong văn bản do AI tạo ra. "
        "Bạn phải thực hiện phân tích theo từng bước logic và đưa ra nhãn cuối cùng là một trong ba loại: 'no', 'intrinsic', hoặc 'extrinsic'."
    )

    user_message = (
        f"Hãy thực hiện phân tích chi tiết cho dữ liệu sau đây.\n\n"
        f"Ngữ cảnh:\n{sample['context']}\n\n"
        f"Câu hỏi:\n{sample['prompt']}\n\n"
        f"Phản hồi cần đánh giá:\n{sample['response']}"
    )

    label = sample['label']
    if label == 'no':
        analysis = (
            "Phân tích từng bước:\n"
            "1. Phân tích mâu thuẫn (Intrinsic): Phản hồi không chứa bất kỳ thông tin nào mâu thuẫn trực tiếp với Ngữ cảnh.\n"
            "2. Phân tích thông tin ngoài (Extrinsic): Phản hồi không bổ sung thông tin, chi tiết hay sự kiện nào không có trong Ngữ cảnh. Toàn bộ thông tin đều có thể được suy ra từ Ngữ cảnh.\n"
            "3. Kết luận: Vì phản hồi hoàn toàn trung thực và bám sát Ngữ cảnh để trả lời Câu hỏi, nhãn phù hợp là 'no'."
        )
    elif label == 'intrinsic':
        analysis = (
            "Phân tích từng bước:\n"
            "1. Phân tích mâu thuẫn (Intrinsic): Phản hồi chứa thông tin mâu thuẫn hoặc làm sai lệch chi tiết có trong Ngữ cảnh.\n"
            "2. Phân tích thông tin ngoài (Extrinsic): Không xét đến vì đã phát hiện mâu thuẫn.\n"
            "3. Kết luận: Do có sự mâu thuẫn trực tiếp với Ngữ cảnh, nhãn phù hợp là 'intrinsic'."
        )
    else:  # 'extrinsic'
        analysis = (
            "Phân tích từng bước:\n"
            "1. Phân tích mâu thuẫn (Intrinsic): Phản hồi không chứa thông tin nào mâu thuẫn trực tiếp với Ngữ cảnh.\n"
            "2. Phân tích thông tin ngoài (Extrinsic): Phản hồi đã bổ sung thêm các chi tiết, sự kiện hoặc bối cảnh không thể tìm thấy hoặc suy luận từ Ngữ cảnh.\n"
            "3. Kết luận: Vì phản hồi đã thêm thông tin không có căn cứ từ Ngữ cảnh, nhãn phù hợp là 'extrinsic'."
        )

    assistant_message = f"{analysis}\n\nNhãn cuối cùng:\n{label}"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message},
    ]

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def create_inference_prompt(sample, tokenizer):
    """Build a prompt for inference using the tokenizer's chat template."""

    system_message = (
        "Bạn là một chuyên gia kiểm định thông tin, cực kỳ cẩn thận và chính xác. "
        "Nhiệm vụ của bạn là phân tích và phát hiện ảo giác (hallucination) trong văn bản do AI tạo ra. "
        "Bạn phải thực hiện phân tích theo từng bước logic và đưa ra nhãn cuối cùng là một trong ba loại: 'no', 'intrinsic', hoặc 'extrinsic'."
    )

    user_message = (
        f"Hãy thực hiện phân tích chi tiết cho dữ liệu sau đây.\n\n"
        f"Ngữ cảnh:\n{sample['context']}\n\n"
        f"Câu hỏi:\n{sample['prompt']}\n\n"
        f"Phản hồi cần đánh giá:\n{sample['response']}"
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def apply_prompt_template(row, tokenizer):
    sample = {
        'context': row['context'],
        'prompt': row['prompt'],
        'response': row['response'],
        'label': row['label'],
    }
    return create_finetuning_prompt(sample, tokenizer)

