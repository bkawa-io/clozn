"""Small deterministic template helpers for engine-client test doubles."""


def render_qwen(messages):
    system = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    for message in messages:
        if message.get("role") == "system" and message.get("content"):
            system = message["content"]
    prompt = f"<|im_start|>system\n{system}<|im_end|>\n"
    for message in messages:
        if message.get("role") in ("user", "assistant"):
            prompt += (
                f"<|im_start|>{message['role']}\n"
                f"{message.get('content', '')}<|im_end|>\n"
            )
    return prompt + "<|im_start|>assistant\n"
