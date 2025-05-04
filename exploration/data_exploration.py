import re
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch import nn
import torch
import pandas as pd
from datasets import Dataset
from utils.preprocessing import get_parsed_data, get_data
import seaborn as sns
import os
import pronouncing
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# TODO: check about the weird token in the first song in get_data
data = get_parsed_data(number_of_songs=20000)

# 1, number of words (Shaked)
def number_of_words_plot(data):
    verse_wc = []
    chorus_wc = []
    for key in data.keys():
        song = data[key]
        for part in song:
            part_split = re.split(r'[ \n]',  part)
            if song[part] == "verse":
                verse_wc.append(len(part_split))
            elif song[part] == "chorus":
                chorus_wc.append(len(part_split))
            else:
                continue

    print(len(verse_wc))
    print(len(chorus_wc))
    plt.hist(verse_wc, bins=50, color='lightcoral', alpha=0.6, label='verse', density=True)
    plt.hist(chorus_wc, bins=50, color='skyblue', alpha=0.6, label='chorus', density=True)
    plt.xlabel("Amount of Words in Part")
    plt.ylabel("Frequency")
    plt.title("Amount of Words")
    plt.legend()
    plt.savefig("../plots/word_count_hist.png")
    # plt.show()


def number_of_row_plot(data):
    verse_rows_count = []
    chorus_rows_count = []
    for key in data.keys():
        song = data[key]
        for part in song:
            part_split = re.split(r'[\n]',  part)
            if song[part] == "verse":
                verse_rows_count.append(len(part_split))
            elif song[part] == "chorus":
                chorus_rows_count.append(len(part_split))
            else:
                continue

    print(len(verse_rows_count))
    print(len(chorus_rows_count))
    plt.hist(verse_rows_count, bins=50, color='lightcoral', alpha=0.6, label='verse', density=True)
    plt.hist(chorus_rows_count, bins=50, color='skyblue', alpha=0.6, label='chorus', density=True)
    plt.xlabel("Amount of Rows in Part")
    plt.ylabel("Frequency")
    plt.title("Amount of Rows")
    plt.legend()
    # os.makedirs("../plots", exist_ok=True)
    plt.savefig("../plots/rows_count_hist.png")
    # plt.show()


def calculate_precision_and_recall(predictions, labels):
    """
    Calculate precision and recall for the given predictions and labels.
    :param predictions: List of predicted labels
    :param labels: List of true labels
    :return: Tuple of (precision, recall)
    """
    true_positives = sum(p == l == 1 for p, l in zip(predictions, labels))
    false_positives = sum(p == 1 and l == 0 for p, l in zip(predictions, labels))
    false_negatives = sum(p == 0 and l == 1 for p, l in zip(predictions, labels))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return precision, recall

def load_predictions_and_labels(test_data_path, predictions_path):
    # Load the dataset from prequarters
    test_data = pd.read_parquet(test_data_path)
    true_labels = test_data['label'].tolist()  # Assuming 'label' is the column with true labels
    predictions = pd.read_parquet(predictions_path)['predictions'].to_list()  # Assuming predictions are in a column
    return predictions, true_labels

def run_precision_and_recall():
    predictions, true_labels = load_predictions_and_labels("./data/test_data.parquet", "./data/test_predictions.parquet")
    precision, recall = calculate_precision_and_recall(predictions, true_labels)
    print(f"Precision: {precision}, Recall: {recall}")


def unique_word_ratio(paragraph):
    """
    Calculate the ratio of unique words to total words in a paragraph.

    Args:
        paragraph (str): Text to analyze, may include newlines

    Returns:
        float: Ratio of unique words to total words (0.0 to 1.0)
    """
    # Handle empty input
    if not paragraph or paragraph.isspace():
        return 0.0

    # Split into words and remove punctuation and convert to lowercase
    words = paragraph.lower().split()

    # Clean words (remove punctuation)
    cleaned_words = []
    for word in words:
        cleaned_word = ''.join(char for char in word if char.isalnum() or char == "'")
        if cleaned_word:  # Only add non-empty strings
            cleaned_words.append(cleaned_word)

    # Count total words and unique words
    total_words = len(cleaned_words)
    unique_words = len(set(cleaned_words))

    # Calculate ratio
    if total_words == 0:
        return 0.0

    return unique_words / total_words

def unigram_repetition_score(text):
    # Lowercase and split text into words (basic cleaning)
    words = re.findall(r'\b\w+\b', text.lower())

    if not words:
        return 0.0  # Avoid division by zero

    unique_words = set(words)
    total_words = len(words)
    repetitiveness = 1 - len(unique_words) / total_words

    return repetitiveness


def bigram_repetition_score(text):
    # Lowercase and split text into words
    words = re.findall(r'\b\w+\b', text.lower())

    if len(words) < 2:
        return 0.0  # No bigrams possible

    # Create bigrams
    bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]

    total_bigrams = len(bigrams)
    unique_bigrams = len(set(bigrams))

    repetitiveness = 1 - unique_bigrams / total_bigrams

    return repetitiveness

def calculate_repetition_score(data):
    chorus_unigram, verse_unigram, chorus_bigram, verse_bigram = [], [], [], []

    for v in data.values():
        for lyrics, song_part in v.items():
            if song_part == 'chorus':
                chorus_unigram.append(unigram_repetition_score(lyrics))
                chorus_bigram.append(bigram_repetition_score(lyrics))
            if song_part == 'verse':
                verse_unigram.append(unigram_repetition_score(lyrics))
                verse_bigram.append(bigram_repetition_score(lyrics))

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Unigram repetition plot
    axes[0].hist(np.array(chorus_unigram)*100, bins=20, alpha=0.7, label='Chorus', color='skyblue', density=True)
    axes[0].hist(np.array(verse_unigram)*100, bins=20, alpha=0.7, label='Verse', color='lightcoral', density=True)
    axes[0].set_title('Unigram Repetition (Normalized)')
    axes[0].set_xlabel('Repetition Score')
    axes[0].set_ylabel('Density')
    axes[0].legend()

    # Bigram repetition plot
    axes[1].hist(np.array(chorus_bigram)*100, bins=20, alpha=0.7, label='Chorus', color='skyblue', density=True)
    axes[1].hist(np.array(verse_bigram)*100, bins=20, alpha=0.7, label='Verse', color='lightcoral', density=True)
    axes[1].set_title('Bigram Repetition (Normalized)')
    axes[1].set_xlabel('Repetition Score')
    axes[1].set_ylabel('Density')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("../plots/repetitiveness_score.png")
    plt.close(fig)

def calculate_unique_words(data):
    chorus_data, verse_data =[], []
    for v in data.values():
        for lyrics, song_part in v.items():
            if song_part == 'chorus':
                chorus_data.append(unique_word_ratio(lyrics)*100)
            if song_part == 'verse':
                verse_data.append(unique_word_ratio(lyrics)*100)

    plt.hist(verse_data, bins=50, color='lightcoral', alpha=0.6, label='verse', density=True)
    plt.hist(chorus_data, bins=50, color='skyblue', alpha=0.6, label='chorus', density=True)
    plt.xlabel("Ratio of Unique Words")
    plt.ylabel("Frequency")
    # plt.title("Histogram of the Rows in Verses")
    plt.legend()
    os.makedirs("../plots", exist_ok=True)
    plt.savefig("../plots/unique_words_hist.png")


def rhyme_density(text):
    def words_rhyme(word1, word2):
        rhymes = pronouncing.rhymes(word1.lower())
        return word2.lower() in rhymes
    lines = text.strip().split('\n')
    endings = [line.strip().split()[-1] for line in lines if line.strip()]

    rhyme_count = 0
    total_pairs = 0

    for i in range(len(endings) - 1):
        if words_rhyme(endings[i], endings[i + 1]):
            rhyme_count += 1
        total_pairs += 1

    if total_pairs == 0:
        return 0.0
    return rhyme_count / total_pairs

def calculate_rhyme_avg(data):
    chorus_data, verse_data = [], []
    for v in data.values():
        for lyrics, song_part in v.items():
            if song_part == 'chorus':
                chorus_data.append(rhyme_density(lyrics))
            if song_part == 'verse':
                verse_data.append(rhyme_density(lyrics))

    plt.hist(np.array(chorus_data)*100, bins=50, alpha=0.7, label='Chorus', color='skyblue', density=True)
    plt.hist(np.array(verse_data)*100, bins=50, alpha=0.7, label='Verse', color='lightcoral', density=True)  # Removed 'label'

    # Set axis limits
    # plt.xlim(0, 1)
    # plt.ylim(0, 40)

    # Title and labels
    plt.title('Rhyming Percentages')
    plt.xlabel('Inter-part Rhyming Ratio')
    plt.ylabel('Frequency')

    # Add legend (it will show only 'Chorus' and 'Verse')
    plt.legend()

    # Save the plot
    plt.savefig("../plots/rhyming_score.png")

if __name__ == '__main__':
    # 8. precision and recall (Tomer)
    # run_precision_and_recall()
    # number_of_words_plot(data)
    # number_of_row_plot(data)
    calculate_repetition_score(data)
    # calculate_rhyme_avg(data)

# 2. nuber of tokens (Shaked)

# 3. number of sentences/rows (Shaked)
# 4. sub-repetitions in the corus (Ariel)
# 5. repetitions of words in the paragraphs(yeah yeah yeah) (Ariel)
# 6. rhymes checking (Ariel)
# 7. number of unique words (Shaked)
# 9. single chorus in the song (Tomer)



