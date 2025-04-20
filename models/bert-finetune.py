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

class CustomBertModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(CustomBertModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.to(self.device)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

class BertFineTuner:
    def __init__(self, model_name, num_labels, learning_rate=5e-5, batch_size=32, epochs=10):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = CustomBertModel(model_name, num_labels)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)

    def prepare_data(self, train_data, test_data):
        #make example dataset:
        train_dataset = Dataset.from_pandas(train_data)
        test_dataset = Dataset.from_pandas(test_data)

        train_dataset = train_dataset.map(lambda x: self.tokenizer(x['text'], truncation=True, padding='max_length', max_length=512), batched=True)
        test_dataset = test_dataset.map(lambda x: self.tokenizer(x['text'], truncation=True, padding='max_length', max_length=512), batched=True)

        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        return DataLoader(train_dataset, batch_size=self.batch_size, pin_memory=True), DataLoader(test_dataset, batch_size=self.batch_size, pin_memory=True)

    def train(self, train_loader):
        self.model.train()
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

    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

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
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

        avg_loss = total_loss / len(test_loader)
        accuracy = correct_predictions / total_predictions
        print(f"Test Loss: {avg_loss}, Accuracy: {accuracy}")

if __name__ == "__main__":
    model_name = 'bert-base-uncased'
    num_labels = 2
    dataset_name = 'mrYou/Lyrics_eng_dataset'

    bert_fine_tuner = BertFineTuner(model_name, num_labels)
    data = get_data()
    # split the data into train and test sets
    train_data = data.sample(frac=0.8, random_state=42)
    test_data = data.drop(train_data.index)
    train_loader, test_loader = bert_fine_tuner.prepare_data(train_data, test_data)

    bert_fine_tuner.train(train_loader)
    bert_fine_tuner.evaluate(test_loader)

    # Save the model
    bert_fine_tuner.model.save_pretrained('./fine_tuned_bert')
    bert_fine_tuner.tokenizer.save_pretrained('./fine_tuned_bert')

