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
python3 ./train.py --precision bf16 --epochs 100 --pos_encoding rope --batch_size 8
```

Optimizations, used for reduced memory consumption:
1. Flash-attention
2. Using BF16 training
3. Activation checkpointing (prevents storing activation tensors inside block, but increases compute complexity)

As a result, around 1.5B model fits into single A5000 GPU for training (each parameter results into more than 1 BF16 value for training, + some activations are stored).

Model architecutre is multiple blocks, each block is attention + MoE, with top-K routing.


Resume training:
```
python3 ./train.py --precision bf16 --pos_encoding rope --batch_size 8 --resume_from latest --epochs 15
```