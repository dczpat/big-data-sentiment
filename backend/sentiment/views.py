from django.shortcuts import render
from .lstm import predict_text
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status

# Create your views here.


@api_view(["GET", "POST"])
def index(request):
    # request should be an HTTP request
    # which may contain several fields
    # right now is just sentence(str) and method(str)
    # print("11111111", request)
    if request.method == "POST":  # or request.method == "OPTIONS":
        # print(dict(request.POST))
        print(request.data)
        sentence = request.data.get("sentence", "happy")
        method = request.data.get("method", "lstm")
        print(sentence, method)
        if method == "lstm" or method == "LSTM":
            res = predict_text(sentence)
            print(res)
        elif method == "bert":
            # res=func(sentence)
            pass
        return Response(res)
    else:
        pass
