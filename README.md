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


### Training custom model

Intitial training start:
```
python3 ./train.py --precision bf16 --epoch 100 --pos_encoding rope --batch_size 8
```

Resume training:
```
python3 ./train.py --precision bf16 --pos_encoding rope --batch_size 8 --resume_from latest --epochs 15
```