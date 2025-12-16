import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = "../hf_model"

tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
if tok.pad_token_id is None and tok.eos_token_id is not None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="cuda"
).eval()

prompt = "for(int i = 0; i < "
inputs = tok(prompt, return_tensors="pt").to(model.device)

with torch.inference_mode():
    out = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.6,
        top_p=0.85,
        repetition_penalty=1.15,
        use_cache=False  # smoke test first
    )

print(tok.decode(out[0], skip_special_tokens=True))
