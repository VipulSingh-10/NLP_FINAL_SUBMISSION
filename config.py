"""
Configuration file for the emotion detection project.
Contains all paths, parameters, and constants.
"""

# Output File Paths
MODEL_FILE_PATH = 'saved2/model.pt'
THRESHOLD_FILE_PATH = 'saved2/thresholds.npy'
PRE_PROCESSED_FILE_PATH = 'saved2/track-a-processed.csv'  # This is just for reference

# File Paths
TRAIN_FILE_PATH = 'data/track-a.csv'  # now expects track-a.csv in project root
EVAL_FILE_PATH = 'data/eng.csv'  # now expects eng.csv in project root
NRC_LEXICON_FILE_PATH = 'data/nrc_lexicon.txt'

NLTK_DATA_DIR = "./nltk_data"  # You can change this to any writable path

# Labels
LABEL_TEXT = 'text'
LABEL_EMOTIONS = ['anger', 'fear', 'joy', 'sadness', 'surprise']

# Preprocessing
NUM_AUGMENTATIONS = 2  # How many sentences created by substituting synonyms for per text

# Model training parameters
DISTILBERT_VERSION = "distilbert-base-uncased"
BERT_MAX_LENGTH = 256
RANDOM_SEED = 42
BERT_BATCH_SIZE = 8
BERT_EPOCHS = 3