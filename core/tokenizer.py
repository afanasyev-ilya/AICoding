
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import os

class BPETokenizer:
    def __init__(self, tokenizer_path: str = None):
        """
        If tokenizer_path exists, loads it. Otherwise call train(...) first.
        """
        if tokenizer_path is not None and os.path.exists(tokenizer_path):
            self.tk = Tokenizer.from_file(tokenizer_path)
        else:
            self.tk = None  # call train() to create

        self._update_special_ids()

    def _update_special_ids(self):
        if self.tk is None:
            self.pad_id = self.bos_id = self.eos_id = None
            return
        self.pad_id = self.tk.token_to_id("<pad>")
        self.bos_id = self.tk.token_to_id("<bos>")
        self.eos_id = self.tk.token_to_id("<eos>")

    @property
    def vocab_size(self):
        if self.tk is None:
            raise ValueError("Tokenizer not initialized. Call train() or load a file.")
        return self.tk.get_vocab_size()

    def train(self, files=None, dataset=None, vocab_size=32000, min_freq=2, save_path="tokenizer.json", max_samples=1000):
        """
        Train on either files or a dataset.
        """
        self.tk = Tokenizer(BPE(unk_token=None))
        self.tk.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self.tk.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_freq,
            special_tokens=["<pad>", "<bos>", "<eos>"]
        )
        
        if files is not None:
            # Original file-based training
            self.tk.train(files=files, trainer=trainer)
        elif dataset is not None:
            # New dataset-based training
            self._train_from_dataset(dataset, trainer, max_samples)
        else:
            raise ValueError("Either 'files' or 'dataset' must be provided")
            
        self.tk.save(save_path)
        self._update_special_ids()
        return save_path

    def _train_from_dataset(self, dataset, trainer, max_samples=1000):
        """Train tokenizer from a Hugging Face dataset"""
        def batch_iterator(batch_size=1000):
            samples_processed = 0
            for example in dataset:
                if samples_processed >= max_samples:
                    break
                yield example["content"]
                samples_processed += 1
        
        # Train using the iterator
        self.tk.train_from_iterator(
            batch_iterator(), 
            trainer=trainer, 
            length=max_samples  # Helps with progress reporting
        )

    def encode(self, s: str, add_bos=False, add_eos=False):
        ids = self.tk.encode(s).ids
        if add_bos and self.bos_id is not None:
            ids = [self.bos_id] + ids
        if add_eos and self.eos_id is not None:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids):
        return self.tk.decode(ids)

