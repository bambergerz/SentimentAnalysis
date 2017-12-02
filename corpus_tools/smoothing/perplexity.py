from corpus_tools.corpus import Corpus
import math

def perplexity_bigram(sentence, probability):
    """
    param sentence: an array of strings representing the words in order of
                    the sentence
    param probability: a function that calculates the conditional probability
                        of arg1 appearing in the sentence right after arg2
                        for the bigram model
    """
    accumulator = 0.0
    sentence = sentence.split(" ")
    length = len(sentence)
    for i in range(1, length):
        # Probability(W_i | W_(i-1))
        #print sentence[i]
        #print sentence[i-1]
        accumulator -= math.log(probability(sentence[i], sentence[i-1]))
    return math.pow(accumulator, (1.0/length))


def perplexity_unigram(sentence, probability):
    """
    param sentence: an array of strings representing the words in order of
                    the sentence
    param probability: a function that calculates the probability of 
                        arg1 appearing in the sentence in the unigram model
    """
    accumulator = 0.0
    sentence = sentence.split(" ")
    length = len(sentence)
    for i in range(1, length):
        # Probability(W_i)
        accumulator -= math.log(probability(sentence[i]))
    return math.pow(accumulator, (1.0/length))
