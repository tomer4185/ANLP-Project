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
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Set your seed
seed = 42
g = torch.Generator()
g.manual_seed(seed)

# Dummy dataset
x = torch.randn(100, 2)
dataset = TensorDataset(x)
class CustomBertModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(CustomBertModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.to(self.device)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def save_pretrained(self, save_directory):
        self.bert.save_pretrained(save_directory)

class BertFineTuner:
    def __init__(self, model_name, num_labels, learning_rate=5e-5, batch_size=32, epochs=10):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = CustomBertModel(model_name, num_labels)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)

    def prepare_data(self, train_data, test_data, use_context=True):
        #make example dataset:
        train_dataset = Dataset.from_pandas(train_data)
        test_dataset = Dataset.from_pandas(test_data)

        if use_context:
            train_dataset = train_dataset.map(lambda x: self.tokenizer(x['text'], x['context'], truncation='only_second', padding='max_length', max_length=512), batched=True)
            test_dataset = test_dataset.map(lambda x: self.tokenizer(x['text'], x['context'], truncation='only_second', padding='max_length', max_length=512), batched=True)
        else:
            train_dataset = train_dataset.map(
                lambda x: self.tokenizer(x['text'], truncation=True, padding='max_length',
                                         max_length=512), batched=True)
            test_dataset = test_dataset.map(
                lambda x: self.tokenizer(x['text'], truncation=True, padding='max_length',
                                         max_length=512), batched=True)

        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        return DataLoader(train_dataset, batch_size=self.batch_size, pin_memory=True, shuffle=True, generator=g), DataLoader(test_dataset, batch_size=self.batch_size, pin_memory=True)

    def train(self, train_loader):
        self.model.train()
        losses =[]
        for epoch in range(self.epochs):
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                labels = batch['label'].to(self.model.device)

                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()

                print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
                losses.append(loss.item())
        plt.scatter(x = losses, y = [i+1 for i in range(len(losses))])
        plt.show()

    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        all_predictions = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                labels = batch['label'].to(self.model.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.append(predictions.cpu())
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

        avg_loss = total_loss / len(test_loader)
        accuracy = correct_predictions / total_predictions
        print(f"Test Loss: {avg_loss}, Accuracy: {accuracy}")
        # save the predictions to df
        predictions = torch.cat(all_predictions).numpy()
        test_df = pd.DataFrame(predictions, columns=["predictions"])
        test_df.to_parquet("./data/test_predictions.parquet")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_context", action="store_true", help="Enable context using")
    # Add other arguments here as needed
    parser.add_argument("--output_dir", type=Path, default="../checkpoints/longformer")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    no_context = args.no_context
    model_name = 'bert-base-uncased'
    num_labels = 2
    dataset_name = 'mrYou/Lyrics_eng_dataset'

    bert_fine_tuner = BertFineTuner(model_name, num_labels)
    data = get_data()
    # split the data into train and test sets
    train_data = data.sample(frac=0.8, random_state=42)
    test_data = data.drop(train_data.index)
    # save the data to parquet
    if not os.path.exists('./data'):
        os.makedirs('./data')
    train_data.to_parquet("./data/train_data.parquet")
    test_data.to_parquet("./data/test_data.parquet")
    train_loader, test_loader = bert_fine_tuner.prepare_data(train_data, test_data, use_context= not no_context )

    bert_fine_tuner.train(train_loader)
    bert_fine_tuner.evaluate(test_loader)

    # Save the model
    # create the directory if it does not exist
    if not os.path.exists('./Results'):
        os.makedirs('./Results')
    bert_fine_tuner.model.save_pretrained('./Results/fine_tuned_bert')
    bert_fine_tuner.tokenizer.save_pretrained('./Results/fine_tuned_bert')

    ############################################################################

    # Load the model and tokenizer
    # Specify the directory where the model and tokenizer were saved
    # save_directory = './Results/fine_tuned_bert'
    #
    # # Load the tokenizer
    # tokenizer = BertTokenizer.from_pretrained(save_directory)
    #
    # # Load the model
    # model = BertForSequenceClassification.from_pretrained(save_directory)
    #
    # # Example usage
    # text = "This is a test sentence."
    # inputs = tokenizer(text, return_tensors="pt", truncation=True,
    #                    padding="max_length", max_length=512)
    # outputs = model(**inputs)
    # print(outputs)

