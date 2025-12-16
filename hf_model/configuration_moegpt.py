from transformers import PretrainedConfig

class MoEGPTConfig(PretrainedConfig):
    model_type = "moegpt"

    def __init__(self, **kwargs):
        # All fields come from config.json.
        # PretrainedConfig will store unknown keys as attributes automatically.
        super().__init__(**kwargs)

        # --- HF/SGLang compatibility aliases ---
        # hidden_size
        if not hasattr(self, "hidden_size"):
            if hasattr(self, "n_embd"):
                self.hidden_size = self.n_embd
            elif hasattr(self, "d_model"):
                self.hidden_size = self.d_model

        # attention heads / layers
        if not hasattr(self, "num_attention_heads") and hasattr(self, "n_head"):
            self.num_attention_heads = self.n_head

        if not hasattr(self, "num_hidden_layers") and hasattr(self, "n_layer"):
            self.num_hidden_layers = self.n_layer

        # max positions
        if not hasattr(self, "max_position_embeddings") and hasattr(self, "block_size"):
            self.max_position_embeddings = self.block_size

        # ---- NEW: KV heads (MHA/GQA/MQA) ----
        # For your current attention (same number of Q/K/V heads), this should match num_attention_heads.
        if not hasattr(self, "num_key_value_heads"):
            if hasattr(self, "num_attention_heads"):
                self.num_key_value_heads = int(self.num_attention_heads)
            elif hasattr(self, "n_head"):
                self.num_key_value_heads = int(self.n_head)
            else:
                # last-resort safe default
                self.num_key_value_heads = 1

