import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class SongPartClassifier(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', hidden_size=768,
                 num_labels=2, num_transformer_layers=2):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8),
            num_layers=num_transformer_layers
        )
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, parts):
        # parts: list of N strings (the parts of the song)
        inputs = self.tokenizer(parts, return_tensors='pt', padding=True,
                                truncation=True)
        outputs = self.bert(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0,
                         :]  # shape: (N, hidden_size)

        # Add sequence dimension for Transformer: (seq_len=N, batch=1, hidden)
        transformer_input = cls_embeddings.unsqueeze(1)
        transformer_output = self.transformer(
            transformer_input)  # shape: (N, 1, hidden_size)
        logits = self.classifier(
            transformer_output.squeeze(1))  # shape: (N, num_labels)

        return logits  # Apply softmax outside if needed
