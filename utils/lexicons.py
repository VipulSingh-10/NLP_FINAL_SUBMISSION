import string
from collections import defaultdict

import numpy as np
import torch
from nltk.corpus import stopwords

from config import NRC_LEXICON_FILE_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_nrc_lexicon(filepath):
    lexicon = defaultdict(lambda: [0] * 5)
    emotion_index = {"anger": 0, "fear": 1, "joy": 2, "sadness": 3, "surprise": 4}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            word, emotion, value = line.strip().split('\t')
            if emotion in emotion_index and value == "1":
                lexicon[word][emotion_index[emotion]] = 1
    return lexicon


def extract_features(texts_column):
    stop_words = set(stopwords.words("english"))
    lex_dict = load_nrc_lexicon(NRC_LEXICON_FILE_PATH)

    char_lens = torch.tensor([len(x) for x in texts_column], dtype=torch.float32).unsqueeze(1)
    word_lens = torch.tensor([len(x.split()) for x in texts_column], dtype=torch.float32).unsqueeze(1)
    handcrafted = torch.cat([char_lens, word_lens], dim=1).to(device)

    lexicon = np.zeros((len(texts_column), 5))
    for i, text in enumerate(texts_column):
        tokens = [t for t in text.lower().translate(str.maketrans('', '', string.punctuation)).split() if t not in stop_words]
        vec = np.zeros(5)
        match = 0
        for t in tokens:
            if t in lex_dict:
                vec += np.array(lex_dict[t])
                match += 1
        if match:
            vec /= match
        lexicon[i] = vec * 0.5

    lexicon_tensor = torch.tensor(lexicon, dtype=torch.float32).to(device)
    return handcrafted, lexicon_tensor
