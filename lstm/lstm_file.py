#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 16:56:48 2022

@author: zhusisi
"""

import sys
import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
import re
import pickle
from torch.utils.data import TensorDataset, DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SentimentRNN(nn.Module):
    def __init__(self,no_layers,vocab_size,output_dim, hidden_dim,embedding_dim,drop_prob=0.5):
        super(SentimentRNN,self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.no_layers = no_layers
        self.vocab_size = vocab_size
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        #lstm
        self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=self.hidden_dim,
                           num_layers=no_layers, batch_first=True)
        # dropout layer
        self.dropout = nn.Dropout(0.3)
        # linear and sigmoid layer
        self.fc = nn.Linear(self.hidden_dim, output_dim)
        self.sig = nn.Sigmoid()
    def forward(self,x,hidden):
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
        sig_out = sig_out[:, -1] # get last batch of labels
        # return last sigmoid output and hidden state
        return sig_out, hidden
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        h0 = torch.zeros((self.no_layers,batch_size,self.hidden_dim)).to(device)
        c0 = torch.zeros((self.no_layers,batch_size,self.hidden_dim)).to(device)
        hidden = (h0,c0)
        return hidden
   
        
def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    s = re.sub(r"\d", '', s)
    return s

def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

def get_key(val, dic):
    for key, value in dic.items():
        if val == value:
            return key
        
def predict_text(file_name):
    no_layers = 2
    vocab_size = 1000 + 1 #extra 1 for padding
    embedding_dim = 64
    hidden_dim = 256
    output_dim = 1
    model = SentimentRNN(no_layers,vocab_size,output_dim, hidden_dim,embedding_dim,drop_prob=0.5)
    model.load_state_dict(torch.load('state_dict_model.pt', map_location=device))
    model.to(device)
    a_file = open("vocab.pkl", "rb")
    vocab = pickle.load(a_file)
    df = pd.read_csv(file_name+'.csv', index_col=0)
    data_val = df['Tweet_Clean'].values
    input_list = []
    for sent in data_val:
      input_list.append([vocab[preprocess_string(word)] for word in str(sent).lower().split() 
                                        if preprocess_string(word) in vocab.keys()])
    input_arr = np.array(input_list)
    input_list_pad = padding_(input_arr,27)
    input_data = TensorDataset(torch.from_numpy(input_list_pad))
    batch_size = 1
    input_loader = DataLoader(input_data, shuffle=False, batch_size=batch_size)
    prob_list = []
    result_list = []
    input_h = model.init_hidden(batch_size)
    for inputs in input_loader:
      input = inputs[0].to(device)
      input_h = tuple([each.data for each in input_h])
      output, result = model(input, input_h)
      prob_list.append(output.item())
      result = 1 if output.item()>0.5 else 0
      result_list.append(result)
    result_df = pd.DataFrame(list(zip(prob_list, result_list)), columns=['prob', 'result'])
    output_df = pd.concat([df, result_df], axis = 1)
    output_df.to_csv(file_name + ' Outcome.csv')
    return output_df
    
