import re

from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
bert_vocab = set(tokenizer.vocab.keys())


def clean_text(text):
    text = text.lower()

    # Replace numbers with "number"
    text = re.sub(r'\d+', 'number', text)

    # Separate punctuation symbols from words (like 'you!' → 'you !') (!, ?, :, ')
    text = re.sub(r'([a-z])([!?:\'])', r'\1 \2', text)

    # Remove all characters except lowercase letters, spaces, '?' and '!'
    text = re.sub(r'[^a-zA-Z\s?!]', '', text)

    # Remove all single-letter words except 'a' and 'i'
    text = re.sub(r'\b(?![ai]\b)[a-z]\b', '', text)

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize and filter
    tokens = text.split()
    cleaned_tokens = [
        word for word in tokens if word in bert_vocab or word in {"number", "!"}
    ]

    return ' '.join(cleaned_tokens)


def clean_data(text_column):
    return [clean_text(text) for text in text_column]
