from transformers import PretrainedConfig

class MoEGPTConfig(PretrainedConfig):
    model_type = "moegpt"

    def __init__(
        self,
        vocab_size=1536,
        n_embd=1024,
        n_layer=6,
        n_head=16,
        block_size=2048,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.block_size = block_size
