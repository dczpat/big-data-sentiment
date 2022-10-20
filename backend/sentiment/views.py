from django.shortcuts import render
from .lstm import predict_text
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status

# Create your views here.


@api_view(["GET", "POST"])
def dummy(request):
    # request should be an HTTP request
    # which may contain several fields
    # right now is just sentence(str) and method(str)
    if request.method == "POST":
        sentence = request.sentence  # TODO: serializer
        method = request.method
        if method == "lstm":
            res = predict_text(sentence)
        elif method == "bert":
            # res=func(sentence)
            pass
        return Response(res)
    else:
        pass
