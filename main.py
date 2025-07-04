import pandas as pd

from config import LABEL_TEXT
from evaluate import evaluate
from preprocess.data_clearing import clean_data


def predict(csv_path):
    # Loading the training data file
    df = pd.read_csv(csv_path)
    text_column = df[LABEL_TEXT].tolist()

    # clean Text before evaluating
    text_column_cleared = clean_data(text_column)

    return evaluate(text_column_cleared)
