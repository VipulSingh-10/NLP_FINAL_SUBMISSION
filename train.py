import os
import nltk
import random
NLTK_DATA = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(NLTK_DATA, exist_ok=True)
nltk.data.path.insert(0, NLTK_DATA)


for pkg in ["punkt", "wordnet", "stopwords","punkt_tab", "averaged_perceptron_tagger","averaged_perceptron_tagger_eng", "omw-1.4"]:
    nltk.download(pkg, download_dir=NLTK_DATA, quiet=True)


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast

from config import *
from objects.dataset import EmotionDataset
from objects.distilbert import DistilBERTEmotion
from preprocess.data_augmenting import augment_data
from preprocess.data_balancing import balance_data
from preprocess.data_clearing import clean_data
from utils.lexicons import load_nrc_lexicon
from utils.nltk_resources import download_nltk_resources

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure the directory exists
os.makedirs(os.path.dirname(PRE_PROCESSED_FILE_PATH), exist_ok=True)


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preprocess(text_column, labels_column):
    texts_col_cleared = clean_data(text_column)

    texts_col_gen, labels_cols_gen = balance_data(texts_col_cleared, labels_column)

    texts_col_augmented, labels_cols_augmented = augment_data(texts_col_cleared, labels_column, NUM_AUGMENTATIONS)

    texts_col_final = texts_col_cleared + texts_col_gen + texts_col_augmented
    labels_cols_final = labels_column + labels_cols_gen + labels_cols_augmented

    return texts_col_final, labels_cols_final


def train(texts_col, labels_cols):
    texts_col_train, texts_col_validation, labels_cols_train, labels_cols_validation = train_test_split(
        texts_col, labels_cols, test_size=0.1, random_state=RANDOM_SEED
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_VERSION)

    nrc_lexicon = load_nrc_lexicon(NRC_LEXICON_FILE_PATH)

    train_dataset = EmotionDataset(texts_col_train, labels_cols_train, tokenizer, nrc_lexicon, max_len=BERT_MAX_LENGTH)
    val_dataset = EmotionDataset(texts_col_validation, labels_cols_validation, tokenizer, nrc_lexicon, max_len=BERT_MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BERT_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BERT_BATCH_SIZE)

    model = DistilBERTEmotion().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    best_val_loss = float("inf")

    for epoch in range(BERT_EPOCHS):
        print(f"\nEpoch {epoch+1}/{BERT_EPOCHS} ===")

        # ----- Training -----
        model.train()
        total_train_loss = 0

        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            handcrafted = batch["handcrafted"].to(device)
            lexicon_feats = batch["lexicon_feats"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, handcrafted, lexicon_feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
                print(f"[Train] Epoch {epoch+1} | Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"[Train] Epoch {epoch+1} Completed | Avg Loss: {avg_train_loss:.4f}")

        # ----- Validation -----
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for j, batch in enumerate(val_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                handcrafted = batch["handcrafted"].to(device)
                lexicon_feats = batch["lexicon_feats"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask, handcrafted, lexicon_feats)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                if (j + 1) % 5 == 0 or (j + 1) == len(val_loader):
                    print(f"[Val] Batch {j+1}/{len(val_loader)} | Loss: {loss.item():.4f}")

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"[Val] Epoch {epoch+1} Completed | Avg Loss: {avg_val_loss:.4f}")

        # ----- Save Best Model -----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_FILE_PATH)
            print(f"New best model saved to {MODEL_FILE_PATH}")

    print("Training complete.")

    threshold_tuning(model, val_loader)


def threshold_tuning(model, val_loader):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            handcrafted = batch["handcrafted"].to(device)
            lexicon_feats = batch["lexicon_feats"].to(device)
            labels = batch["labels"].cpu().numpy()
            preds = model(input_ids, attention_mask, handcrafted, lexicon_feats).cpu().numpy()

            all_labels.extend(labels)
            all_preds.extend(preds)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    optimal_thresholds = []
    for i in range(5):
        best_thresh = 0.5
        best_f1 = 0
        for t in np.arange(0.1, 0.9, 0.01):
            preds_bin = (all_preds[:, i] > t).astype(int)
            score = f1_score(all_labels[:, i], preds_bin)
            if score > best_f1:
                best_f1 = score
                best_thresh = t
        optimal_thresholds.append(best_thresh)

    np.save(THRESHOLD_FILE_PATH, optimal_thresholds)
    print(f"Thresholds Saved As {THRESHOLD_FILE_PATH}")


def main():

    set_seed(RANDOM_SEED)

    download_nltk_resources()

    # Loading the training data file
    df = pd.read_csv(TRAIN_FILE_PATH)
    text_column = df[LABEL_TEXT].tolist()
    labels_column = df[LABEL_EMOTIONS].values.tolist()
    print(f"Train Data Size: {len(text_column)}")

    # Process data (cleaning, balancing, augmenting)
    text_column_processed, labels_column_processed = preprocess(text_column, labels_column)
    print(f"Final Data Size: {len(text_column_processed)}")

    # Training the objects on the processed data
    train(text_column_processed, labels_column_processed)


if __name__ == "__main__":
    main()
