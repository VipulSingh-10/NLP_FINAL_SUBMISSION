import random
from collections import defaultdict

# Define connectors
connectors = [
    'and', 'also', 'moreover', 'furthermore', 'in addition',
    'besides', 'as well as', 'additionally', 'not only that',
    'what’s more', 'on top of that', 'along with that', 'plus',
    'similarly', 'in the same way', 'to add to that'
]

be_verbs = {"am", "is", "are", "was", "were", "be", "being", "been"}
avoid_words = {"i", "number"}


def generate_all_binary_combos(num_emotions):
    return [
        tuple(list(map(int, format(i, f'0{num_emotions}b'))))
        for i in range(1, 2 ** num_emotions)
    ]


def balance_data(texts_col, labels_cols):
    """
    Create synthetic rows by merging texts to reach at least `min_target`
    for all binary label combinations.
    Returns: List of synthetic texts and their corresponding labels.
    """
    num_emotions = len(labels_cols[0])

    # Step 1: Group existing samples by binary label tuple
    combo_to_samples = defaultdict(list)
    for text, label in zip(texts_col, labels_cols):
        combo = tuple(label)
        combo_to_samples[combo].append((text, label))

    # Calculate dynamic min_target from existing distribution
    counts = [len(samples) for samples in combo_to_samples.values()]
    max_count = max(counts)
    min_count = min(counts)
    min_target = (max_count + min_count) // 2

    # Step 2: Generate all possible binary combinations
    all_possible_combos = generate_all_binary_combos(num_emotions)

    synthetic_texts = []
    synthetic_labels = []

    for target_tuple in all_possible_combos:
        current_count = len(combo_to_samples[target_tuple])

        if current_count >= min_target:
            continue  # Already sufficient samples

        # Step 3: Try building this combo from existing partial ones
        needed = min_target - current_count
        partial_sources = [
            combo for combo in combo_to_samples
            if 0 < sum(combo) < sum(target_tuple)
               and len(combo_to_samples[combo]) > 0
               and all(t == 1 or c == 0 for t, c in zip(target_tuple, combo))
        ]

        if len(partial_sources) < 2:
            continue  # Not enough components to synthesize from

        for _ in range(needed):
            # Sample two different partial combos
            combo1, combo2 = random.sample(partial_sources, 2)

            if not combo_to_samples[combo1] or not combo_to_samples[combo2]:
                continue

            sample1 = random.choice(combo_to_samples[combo1])
            sample2 = random.choice(combo_to_samples[combo2])

            text1, label1 = sample1
            text2, label2 = sample2

            conn = random.choice(connectors)
            merged_text = f"{text1} {conn} {text2}"
            merged_label = [max(a, b) for a, b in zip(label1, label2)]

            synthetic_texts.append(merged_text)
            synthetic_labels.append(merged_label)

    return synthetic_texts, synthetic_labels
