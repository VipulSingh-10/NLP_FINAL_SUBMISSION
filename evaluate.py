import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from transformers import (
    DistilBertTokenizerFast
)

from config import *
from objects.distilbert import DistilBERTEmotion
from preprocess.data_clearing import clean_data
from utils.lexicons import extract_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(texts_column):
    model = DistilBERTEmotion().to(device)
    tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_VERSION)
    model.load_state_dict(torch.load(MODEL_FILE_PATH, map_location=device))
    thresholds = np.load(THRESHOLD_FILE_PATH)

    inputs = tokenizer(texts_column, padding=True, truncation=True, return_tensors="pt", max_length=BERT_MAX_LENGTH)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    handcrafted, lexicon_tensor = extract_features(texts_column)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask, handcrafted, lexicon_tensor).cpu().numpy()

    return (outputs > thresholds).astype(int)


def main():
    # Loading the training data file
    df = pd.read_csv(EVAL_FILE_PATH)
    text_column = df[LABEL_TEXT].tolist()
    labels_column = df[LABEL_EMOTIONS].values.tolist()

    # clean Text before evaluating
    texts_col_cleared = clean_data(text_column)

    predictions = evaluate(texts_col_cleared)

    print("\nClassification Report:")
    print(classification_report(labels_column, predictions, target_names=LABEL_EMOTIONS, zero_division=0))


if __name__ == "__main__":
    main()
