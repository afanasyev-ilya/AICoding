### sota contains reference models for python coding (phi-1.3B for now)

Setup venv and make sure transformers are installed:

```
source /home/i.afanasyev/torch-venv/bin/activate
python3 -m pip install transformers
```

Fix huggingface restrictions if needed:
```
export HF_ENDPOINT=https://hf-mirror.com
```

Run sample inference:
```
python3 ./phi.py
```