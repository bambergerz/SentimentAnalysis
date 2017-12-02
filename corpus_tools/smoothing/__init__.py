import os
from corpus_tools.smoothing.simple_smoothing import Smoothing, AddKSmoothing
from corpus_tools.corpus import Corpus

if __name__ == '__main__':
    corpus = Corpus("example")
    addk = AddKSmoothing(corpus, 1)
    print(addk)
    print("unigram", addk.unigram("Hello"))
    print("known bigram", addk.bigram("world", "Hello"))
    print("unknown bigram", addk.bigram("world", "world"))
    print("known unkknown bigram", addk.bigram("Rohit", "world"))
    print("unkown known bigram", addk.bigram("world", "Rohit"))