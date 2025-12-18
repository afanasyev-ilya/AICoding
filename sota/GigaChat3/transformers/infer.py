import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

model_name = "ai-sage/GigaChat3-10B-A1.8B-bf16"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
model.generation_config = GenerationConfig.from_pretrained(model_name)

messages = [
    {"role": "user", "content": "Напиши простейшее матричное умножение на С++."}
]
input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

print("Generation in progress")
outputs = model.generate(input_tensor.to(model.device), max_new_tokens=4000)

result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=False)
print(result)