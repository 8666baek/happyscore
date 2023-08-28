import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import ImageFont

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from streamlit_plotly_events import plotly_events


import pandas as pd
import numpy as np
import torch
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from transformers import AutoModel, AutoTokenizer, AutoConfig

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm, trange


# seed 고정
seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.cuda.manual_seed_all(seed)


class BertRegresser_MultiTarget(nn.Module):
    def __init__(self, base_model, freeze_bert=True):
        super().__init__()
        self.bert = base_model
        self.freeze_bert = freeze_bert
        self.num_targets = num_targets  # Number of individual target values

        self.cls_layer1 = nn.Linear(base_model.config.hidden_size, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.ff1 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        # Output layers for each individual target value
        self.target_layers = nn.ModuleList([nn.Linear(64, 1) for _ in range(10)])

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.last_hidden_state[:, 0, :]
        output = self.cls_layer1(logits)
        output = self.relu1(output)
        output = self.dropout1(output)
        output = self.ff1(output)
        output = self.relu2(output)
        output = self.dropout2(output)

        individual_outputs = [target_layer(output) for target_layer in self.target_layers]
        individual_outputs = torch.cat(individual_outputs, dim=1)

        return individual_outputs

    def freeze_bert_layers(self):
        if self.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

# Initialize the modified model
num_targets = 10  # Number of individual target values


def predict(model, dataloader, device):
    predicted_label = []
    actual_label = []
    model.eval()  # 평가 모드로 설정

    with torch.no_grad():
        for input_ids, attention_mask, target in dataloader:
            input_ids, attention_mask, target = input_ids.to(device), attention_mask.to(device), target.to(device)
            output = model(input_ids, attention_mask)

            predicted_label += output.flatten().cpu().tolist()
            actual_label += target.cpu().tolist()

    model.train()  # 다시 학습 모드로 변경

    return predicted_label, actual_label

class Happy_Dataset_MultiTarget(Dataset):
    def __init__(self, data, maxlen, tokenizer):
        self.df = data.reset_index()
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        excerpt = self.df.loc[index, '일상']
        targets = [0]  # Replace with your column names

        tokens = self.tokenizer.tokenize(excerpt)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        if len(tokens) < self.maxlen:
            tokens += ['[PAD]'] * (self.maxlen - len(tokens))
        else:
            tokens = tokens[:self.maxlen - 1] + ['[SEP]']

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids)
        attention_mask = (input_ids != 0).long()

        targets = torch.tensor(targets, dtype=torch.float32)

        return input_ids, attention_mask, targets

model_nm = 'klue/roberta-small'
base_model = AutoModel.from_pretrained(model_nm)
tokenizer = AutoTokenizer.from_pretrained(model_nm)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

from huggingface_hub import hf_hub_download
model_file = hf_hub_download(repo_id="Juneha/happyscore", filename="multilabel.pth")

#PATH = 'multilabel.pth'
model = BertRegresser_MultiTarget(base_model)
model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')), strict=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def use_model(text):
    df = pd.DataFrame([text], columns=['일상'])
    sample = Happy_Dataset_MultiTarget(data = df, maxlen = 20, tokenizer = tokenizer)
    sample_loader = DataLoader(dataset=sample, batch_size=32, num_workers=1, worker_init_fn=seed_worker,
    generator=g)

    #score = predict(model, sample_loader, device)[0][0]*100
    score = pd.DataFrame(predict(model, sample_loader, device)[0]).T
    score.columns = ['질문1', '질문2', '질문3', '질문4', '질문5', '질문6', '질문7','질문8', '질문9', '질문10']
    score['행복점수'] = score.sum(axis=1)
    score = score * 10
    score = np.round(score, 1)
    return score

st.title('Happiness Predictor')
st.write('Enter a text and find out its happiness score!')

text_input = st.text_input(label='Enter your text:')
if st.button('Predict'):
    if text_input:
        score = use_model(text_input)
        st.write(f'Text: {text_input}')
        st.write(f'Happiness Score: {score["행복점수"].values[0]:.1f}')
        
        st.write(score)

    
    
    
    
    