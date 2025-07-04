import random

from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

# Define connectors
connectors = [
    'and', 'also', 'moreover', 'furthermore', 'in addition',
    'besides', 'as well as', 'additionally', 'not only that',
    'what’s more', 'on top of that', 'along with that', 'plus',
    'similarly', 'in the same way', 'to add to that'
]

be_verbs = {"am", "is", "are", "was", "were", "be", "being", "been"}
avoid_words = {"i", "number"}


def get_wordnet_pos(tag):
    if tag in ["NNP", "NNPS"]:  # Skip proper nouns
        return None
    elif tag.startswith('J'):
        return wordnet.ADJ  # adjective
    elif tag.startswith('V'):
        return wordnet.VERB  # verb
    elif tag.startswith('N'):
        return wordnet.NOUN  # noun
    elif tag.startswith('R'):
        return wordnet.ADV  # adverb
    else:
        return None  # ignore pronouns, determiners, etc.


def get_synonyms(word, pos=wordnet.NOUN):
    synonyms = {word}  # include the original word

    # Get synsets for the given POS
    synsets = wordnet.synsets(word, pos=pos)

    for syn in synsets:
        for lemma in syn.lemmas():
            name = lemma.name().replace('_', ' ')
            # Avoid adding the word itself and duplicates
            if name.lower() != word.lower():
                synonyms.add(name)

    return list(synonyms)


def augment_text(text, num_augmentations=1):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    augmented_texts = []

    for _ in range(num_augmentations):
        new_tokens = []

        for word, tag in tagged:
            wn_pos = get_wordnet_pos(tag)

            if word.lower() not in be_verbs and word.lower() not in avoid_words and wn_pos:
                synonyms = get_synonyms(word, pos=wn_pos)
                word = random.choice(synonyms)

            new_tokens.append(word)

        augmented_texts.append(' '.join(new_tokens))

    return augmented_texts


def augment_data(text_column, label_columns, num_augmentations=1):
    augmented_texts = []
    augmented_labels = []

    for text, label in zip(text_column, label_columns):
        # Add augmented versions
        for aug in augment_text(text, num_augmentations=num_augmentations):
            augmented_texts.append(aug)
            augmented_labels.append(label)

    return augmented_texts, augmented_labels
