
# 1, number of words (Shaked)
# 2. nuber of tokens (Shaked)
# 3. number of sentences/rows (Shaked)
# 4. sub-repetitions in the corus (Ariel)
# 5. repetitions of words in the paragraphs(yeah yeah yeah) (Ariel)
# 6. rhymes checking (Ariel)
# 7. number of unique words (Shaked)
# 8. precision and recall (Tomer)
# 9. single chorus in the song (Tomer)

import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch import nn
import torch
from datasets import Dataset
from utils.preprocessing import get_data

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

if __name__ == '__main__':
    # 8. precision and recall (Tomer)
    run_precision_and_recall()