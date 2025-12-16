import os, sys
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PreTrainedModel, PretrainedConfig
from dataclasses import fields, is_dataclass
from .configuration_moegpt import MoEGPTConfig
from collections import OrderedDict

# Import your original model code from the copied folder
_THIS_DIR = os.path.dirname(__file__)
sys.path.insert(0, _THIS_DIR)
from .moegpt_impl import MoEGPT, MoEGPTConfig as InternalConfig

def _build_internal_config(cfg: PretrainedConfig) -> InternalConfig:
    """
    Convert HF config -> your internal dataclass config.
    No hardcoded dimensions here: everything comes from config.json.
    """
    d = cfg.to_dict()

    # Filter out HF bookkeeping keys and keep only what your dataclass accepts
    if is_dataclass(InternalConfig):
        allowed = {f.name for f in fields(InternalConfig)}
        kwargs = {k: v for k, v in d.items() if k in allowed}
        return InternalConfig(**kwargs)

    # If your InternalConfig is not a dataclass
    return InternalConfig(**d)

class MoEGPTForCausalLM(PreTrainedModel):
    config_class = MoEGPTConfig
    main_input_name = "input_ids"

    # we don't tie any weights, keep it empty.
    _tied_weights_keys = []
    _supports_attention_backend = True

    def __init__(self, config: MoEGPTConfig):
        super().__init__(config)

        # Build your internal config. Map fields as needed.
        internal_cfg = _build_internal_config(config)
        self.model = MoEGPT(internal_cfg)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        past_key_values=None,
        use_cache=None,
        labels=None,
        **kwargs):
        # You can ignore attention_mask/token_type_ids for now
        logits, _, _ = self.model(input_ids)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)

    def prepare_inputs_for_generation(self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        **kwargs):
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    @property
    def all_tied_weights_keys(self):
        # Newer loader code expects a dict-like object with .keys()
        keys = getattr(self, "_tied_weights_keys", None) or []
        return OrderedDict((k, True) for k in keys)

    def get_input_embeddings(self):
        # Adjust attribute names if yours differ.
        return self.model.tok_emb

    def set_input_embeddings(self, value):
        self.model.tok_emb = value

    # Strongly recommended for CausalLMs (some tooling calls these too)
    def get_output_embeddings(self):
        return self.lm_head  # or self.head if that's your name

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings  # or self.head = new_embeddings
