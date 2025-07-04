import torch
import torch.nn as nn
from transformers import DistilBertModel

from config import DISTILBERT_VERSION


class DistilBERTEmotion(nn.Module):
    def __init__(self):
        super(DistilBERTEmotion, self).__init__()
        self.bert = DistilBertModel.from_pretrained(DISTILBERT_VERSION)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(768 + 2 + 5, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.out = nn.Linear(128, 5)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, handcrafted, lexicon_feats):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = out.last_hidden_state[:, 0, :]
        combined = torch.cat((cls_token, handcrafted, lexicon_feats), dim=1)
        x = self.dropout(combined)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        return self.sigmoid(self.out(x))