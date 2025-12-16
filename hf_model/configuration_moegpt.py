from transformers import PretrainedConfig

class MoEGPTConfig(PretrainedConfig):
    model_type = "moegpt"

    def __init__(self, **kwargs):
        # All fields come from config.json.
        # PretrainedConfig will store unknown keys as attributes automatically.
        super().__init__(**kwargs)

