import nltk
import os

from config import NLTK_DATA_DIR  # example: "./nltk_data"


def safe_download(resource_name, path):
    try:
        nltk.data.find(path)
    except LookupError:
        print(f'Downloading {resource_name} to {NLTK_DATA_DIR}...')
        nltk.download(resource_name, download_dir=NLTK_DATA_DIR)


def download_nltk_resources():
    nltk.data.path = [NLTK_DATA_DIR]

    os.makedirs(NLTK_DATA_DIR, exist_ok=True)

    safe_download('punkt', 'tokenizers/punkt')
    safe_download('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger')
    safe_download('stopwords', 'corpora/stopwords')
    safe_download('wordnet', 'corpora/wordnet')
    safe_download('omw-1.4', 'corpora/omw-1.4')
