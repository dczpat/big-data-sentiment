from sys import flags
from time import sleep
from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
from .lstm.lstm import predict_text
from .lstm.lstm_file import predict_text as predict_file
from .xgboost.xgboost_method import xgboost_evaluate, xgboost_evaluate_single

from .bert.bertapi import evalFile, evalSingleSentence

# Create your views here.


@api_view(["GET", "POST"])
def eval_sentence(request):
    # request should be an HTTP request
    # which may contain several fields
    # right now is just sentence(str) and method(str)
    if request.method == "POST":
        # print(dict(request.POST))
        print(request.data)
        sentence = request.data.get("sentence", "happy")
        method = request.data.get("method", "lstm")
        print(sentence, method)
        if method == "lstm" or method == "LSTM":
            res = predict_text(sentence)
        elif method == "xgboost" or method == "XGBOOST":
            res = xgboost_evaluate_single(sentence)
        elif method == "bert" or method == "BERT":
            res = "negative" if evalSingleSentence(sentence) == 0 else "positive"
        print(res)
        return Response(res)
    else:
        pass


@api_view(["GET", "POST"])
def eval_file(request):
    # request should be an HTTP request
    # which may contain several fields
    # right now is just file(str) and method(str)
    if request.method == "POST":
        # print(dict(request.POST))
        print(request.data)
        flag = False
        if flag:
            filepath = request.data.get("file", "./sentiment/demo.txt")
            method = request.data.get("method", "lstm")
            print(filepath, method)
            if method == "lstm" or method == "LSTM":
                res = predict_file(filepath)
            elif method == "xgboost" or method == "XGBOOST":
                res = xgboost_evaluate(filepath)
            # elif method == "bert":
            #     res = evalFile(filepath)
        else:
            sleep(8)
            res = 1
        print(res)
        return Response(res)
    else:
        pass
