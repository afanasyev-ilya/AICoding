import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "microsoft/phi-1_5"  # 1.3B params

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)

# Simple coding task: complete an iterative binary search implementation
prompt = """import bisect

def binary_search(arr, target):
    \"\"\"Return the index of target in sorted list arr, or -1 if not found.
    Use iterative binary search (no recursion).
    \"\"\"
"""

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    # Phi-1.5 model card uses no attention mask in the basic example.:contentReference[oaicite:3]{index=3}
    return_attention_mask=False,
).to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,      # greedy; for code this is often fine
        temperature=0.0,
    )

generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("=== PROMPT + COMPLETION ===")
print(generated)

# (Optional) Extract only the function if the model babbles after it:
def extract_function(text, func_name="binary_search"):
    lines = text.splitlines()
    keep = []
    recording = False
    for ln in lines:
        if ln.strip().startswith(f"def {func_name}("):
            recording = True
        if recording:
            keep.append(ln)
    return "\n".join(keep)

print("\n=== EXTRACTED FUNCTION ===")
print(extract_function(generated))
