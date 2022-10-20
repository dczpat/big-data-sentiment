#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 00:06:25 2022

@author: zhusisi
"""
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import re
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SentimentRNN(nn.Module):
    def __init__(
        self,
        no_layers,
        vocab_size,
        output_dim,
        hidden_dim,
        embedding_dim,
        drop_prob=0.5,
    ):
        super(SentimentRNN, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.no_layers = no_layers
        self.vocab_size = vocab_size
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # lstm
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=no_layers,
            batch_first=True,
        )
        # dropout layer
        self.dropout = nn.Dropout(0.3)
        # linear and sigmoid layer
        self.fc = nn.Linear(self.hidden_dim, output_dim)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        # embeddings and lstm_out
        embeds = self.embedding(x)  # shape: B x S x Feature   since batch = True
        # print(embeds.shape)  #[128, 27, 64]
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        # dropout and fully connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # sigmoid function
        sig_out = self.sig(out)
        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]  # get last batch of labels
        # return last sigmoid output and hidden state
        return sig_out, hidden

    def init_hidden(self, batch_size):
        """Initializes hidden state"""
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        h0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(device)
        c0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(device)
        hidden = (h0, c0)
        return hidden


def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", "", s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", "", s)
    # replace digits with no space
    s = re.sub(r"\d", "", s)
    return s


def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review) :] = np.array(review)[:seq_len]
    return features


def predict_text(text):
    no_layers = 2
    vocab_size = 1000 + 1  # extra 1 for padding
    embedding_dim = 64
    hidden_dim = 256
    output_dim = 1
    model = SentimentRNN(
        no_layers, vocab_size, output_dim, hidden_dim, embedding_dim, drop_prob=0.5
    )
    model.load_state_dict(torch.load("state_dict_model.pt", map_location=device))

    model.to(device)

    a_file = open("vocab.pkl", "rb")
    vocab = pickle.load(a_file)

    word_seq = np.array(
        [
            vocab[preprocess_string(word)]
            for word in text.split()
            if preprocess_string(word) in vocab.keys()
        ]
    )
    word_seq = np.expand_dims(word_seq, axis=0)
    pad = torch.from_numpy(padding_(word_seq, 500))
    inputs = pad.to(device)
    batch_size = 1
    h = model.init_hidden(batch_size)
    h = tuple([each.data for each in h])
    output, h = model(inputs, h)
    prob = output.item()
    status = "positive" if prob > 0.5 else "negative"
    return status


if __name__ == "__main__":
    for _ in range(100):
        print(predict_text("Have a good day!"))
