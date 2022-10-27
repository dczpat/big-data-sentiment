# -*- coding: utf-8 -*-
"""XGBOOST_method.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FpZP6HXwBeSNZJyPq5sE-pJQNM8Ob-J3
"""

import pandas as pd
import numpy as np
import pickle

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

import nltk

# nltk.download('punkt')
# nltk.download('stopwords')
import string as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer
import re
from tqdm.notebook import tqdm_notebook

tqdm_notebook.pandas()

# file = open("/Users/zhaoxinzhu/Desktop/test_group.txt",encoding="utf-16")


def xgboost_evaluate(filename):
    model = XGBClassifier(
        max_depth=4,
        learning_rate=0.15,
        n_estimators=100,
        silent=True,
        objective="binary:logistic",
        booster="gbtree",
        n_jobs=1,
        nthread=None,
        gamma=0,
        min_child_weight=1,
        max_delta_step=0,
        subsample=1,
        colsample_bytree=1,
        colsample_bylevel=1,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        base_score=0.5,
        random_state=0,
        seed=None,
        missing=None,
    )
    file = open(filename, encoding="utf-8")
    input = file.readlines()
    stemmer = LancasterStemmer()
    test_corpus = []
    for i in tqdm_notebook(range(len(input))):
        word = re.sub("((www[^\s]+)|(http[^\s]+))", " ", input[i].lower())
        word = word_tokenize(word)
        word = [
            stemmer.stem(y)
            for y in word
            if y not in (list(stopwords.words("english")) + list(st.punctuation))
        ]
        j = " ".join(word)
        test_corpus.append(j)
    mid = pd.DataFrame(test_corpus)
    mid.columns = ["text"]
    feature_path = "./sentiment/xgboost/feature.pkl"
    loaded_vec = CountVectorizer(
        decode_error="replace", vocabulary=pickle.load(open(feature_path, "rb"))
    )
    # 加载TfidfTransformer
    tfidftransformer_path = "./sentiment/xgboost/tfidftransformer.pkl"
    tfidftransformer = pickle.load(open(tfidftransformer_path, "rb"))
    # 测试用transform，表示测试数据，为list
    test_tfidf = tfidftransformer.transform(loaded_vec.transform(mid["text"]))
    clf = XGBClassifier()
    booster = xgb.Booster()
    booster.load_model("./sentiment/xgboost/xgboost_classifier_model.model")
    clf._Booster = booster
    clf._le = LabelEncoder().fit([1, 0])
    y_mid = clf.predict(test_tfidf, validate_features=False)
    f = open("./sentiment/xgboost/ans.txt", "w")
    for i, line in enumerate(input):
        f.write(line + "\x20" + "\x20")
        status = "positive" if y_mid[i] == 1 else "negative"
        f.write(status + "\n")
    f.close()
    return "./sentiment/xgboost/ans.txt"


# xgboost_evaluate(file)

# import 的包与上面的共用
a = "bad day"


def xgboost_evaluate_single(sentence):
    word = re.sub("((www[^\s]+)|(http[^\s]+))", " ", sentence.lower())
    stemmer = LancasterStemmer()
    test_corpus = []
    word = word_tokenize(word)
    word = [
        stemmer.stem(y)
        for y in word
        if y not in (list(stopwords.words("english")) + list(st.punctuation))
    ]
    j = " ".join(word)
    test_corpus.append(j)
    mid = pd.DataFrame(test_corpus)
    mid.columns = ["text"]
    # tf_idf = TfidfVectorizer()
    # mid_ = tf_idf.fit_transform(mid['text'])
    feature_path = "./sentiment/xgboost/feature.pkl"
    loaded_vec = CountVectorizer(
        decode_error="replace", vocabulary=pickle.load(open(feature_path, "rb"))
    )
    # 加载TfidfTransformer
    tfidftransformer_path = "./sentiment/xgboost/tfidftransformer.pkl"
    tfidftransformer = pickle.load(open(tfidftransformer_path, "rb"))
    # 测试用transform，表示测试数据，为list
    test_tfidf = tfidftransformer.transform(loaded_vec.transform(mid["text"]))
    clf = XGBClassifier()
    booster = xgb.Booster()
    booster.load_model("./sentiment/xgboost/xgboost_classifier_model.model")
    # cols_when_model_builds = booster.feature_names
    clf._Booster = booster
    clf._le = LabelEncoder().fit([1, 0])
    # pd_dataframe = mid_[cols_when_model_builds]
    y_mid = clf.predict(test_tfidf, validate_features=False)
    status = "positive" if y_mid == 1 else "negative"
    return status


# b = xgboost_evaluate_single(a)
# print(b)
