import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch import nn
import torch
from datasets import Dataset

class CustomBertModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(CustomBertModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

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

    def prepare_data(self, dataset_name):
        # dataset = load_dataset(dataset_name)
        # train_dataset = dataset['train']
        # test_dataset = dataset['test']
        train_data = pd.DataFrame([
            {'text': 'This is a dog class.', 'label': 1},
            {'text': 'all dogs are cute', 'label': 1},
            {'text': 'I am loving my new dog, it is a golden retriever', 'label': 1},
            {'text': 'everybody needs a dog', 'label': 1},
            {'text': 'This is a cat class.', 'label': 0},
            {'text': 'cats are leaking themselves', 'label': 0},
            {'text': 'I have a red hair class.', 'label': 0},
            {'text': 'cats should be clean', 'label': 0},
            {'text': 'I want a cat!.', 'label': 0}
        ])
        test_data = pd.DataFrame([
            {'text' : "cat class", 'label': 0},
            {'text' : "dog class", 'label': 1},
            {"text" : "do you like cats?", 'label': 0},
            {"text" : "do you like dogs?", 'label': 1},
            ])
        #make example dataset:
        train_dataset = Dataset.from_pandas(train_data)
        test_dataset = Dataset.from_pandas(test_data)

        train_dataset = train_dataset.map(lambda x: self.tokenizer(x['text'], truncation=True, padding='max_length', max_length=512), batched=True)
        test_dataset = test_dataset.map(lambda x: self.tokenizer(x['text'], truncation=True, padding='max_length', max_length=512), batched=True)

        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        return DataLoader(train_dataset, batch_size=self.batch_size), DataLoader(test_dataset, batch_size=self.batch_size)

    def train(self, train_loader):
        self.model.train()
        for epoch in range(self.epochs):
            for batch in train_loader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['label']

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
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['label']

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
    train_loader, test_loader = bert_fine_tuner.prepare_data(dataset_name)

    bert_fine_tuner.train(train_loader)
    bert_fine_tuner.evaluate(test_loader)

    # Save the model
    # bert_fine_tuner.model.save_pretrained('./fine_tuned_bert')
    # bert_fine_tuner.tokenizer.save_pretrained('./fine_tuned_bert')

