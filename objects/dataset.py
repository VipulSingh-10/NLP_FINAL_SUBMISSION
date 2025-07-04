import numpy as np
import torch
from torch.utils.data import Dataset


class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, nrc_lexicon, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.nrc_lexicon = nrc_lexicon
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        char_len = len(text)
        word_len = len(text.split())
        original_tokens = text.split()

        encodings = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")

        handcrafted = torch.tensor([char_len, word_len], dtype=torch.float32)

        lexicon_scores = np.zeros(5)
        match_count = 0
        for token in original_tokens:
            if token in self.nrc_lexicon:
                lexicon_scores += np.array(self.nrc_lexicon[token])
                match_count += 1
        if match_count > 0:
            lexicon_scores /= match_count
        lexicon_scores *= 0.5
        lexicon_tensor = torch.tensor(lexicon_scores, dtype=torch.float32)

        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "handcrafted": handcrafted,
            "lexicon_feats": lexicon_tensor,
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }