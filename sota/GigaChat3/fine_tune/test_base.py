import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

model_name = "ai-sage/GigaChat3-10B-A1.8B-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
model.generation_config = GenerationConfig.from_pretrained(model_name)

messages = (
    "Ниже я написал подробное доказательство теоремы о неподвижной точке:"
)
inputs = tokenizer(messages, return_tensors="pt").to(model.device)
outputs = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask)

result = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
print(result)
