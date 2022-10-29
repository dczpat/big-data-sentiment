import transformers
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
import torch

import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PRE_TRAINED_MODEL_NAME = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
class_names = ["negative", "positive"]
MAX_LEN = 32


class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(outs["pooler_output"])
        return self.out(output)


class TwitterSentimentDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        text = self.df.text[item]
        target = self.df.target[item]
        no = self.df.no[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            # padding="longest",
            padding=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "targets": torch.tensor(target, dtype=torch.long),
            "no": torch.tensor(no, dtype=torch.long),
        }


def create_data_loader(df, tokenizer, MAX_LEN, batch_size):
    ds = TwitterSentimentDataset(df, tokenizer, MAX_LEN)
    return DataLoader(ds, batch_size, num_workers=4)


PATH = "./sentiment/bert/best_model_state.bin"
model = SentimentClassifier(len(class_names)).to(device)
model.load_state_dict(torch.load(PATH, map_location=device))


def evalSingleSentence(sentence):
    evaluationDF = pd.DataFrame(columns=["target", "text", "no"])
    evaluationDF.loc[len(evaluationDF.index)] = [0, sentence, 0]

    eval_data_loader = create_data_loader(evaluationDF, tokenizer, MAX_LEN, 1)
    with torch.no_grad():
        for d in tqdm(eval_data_loader):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)

            targets = d["targets"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            return preds.item()


def evalFile(filepath):
    df = pd.read_csv(filepath, header=None, encoding="ISO-8859-1")
    df.columns = ["no", "Time", "text"]
    df = df.drop(0)
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    df["target"] = 0

    eval_data_loader = create_data_loader(df, tokenizer, MAX_LEN, 1)

    result = np.zeros(shape=(len(df) + 5))
    with torch.no_grad():
        for d in tqdm(eval_data_loader):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)

            targets = d["targets"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            score = preds.item()
            no = d["no"].item()
            result[no] = score

    df["target"] = result
    df.to_csv(filepath + "_flag.csv")
