import sys
import numpy
import tensorflow

from corpus_tools.smoothing import perplexity

from corpus_tools import corpus
from corpus_tools.smoothing import simple_smoothing

# DEV_PATH = "../Project1/SentimentDataset/Dev/"
# TEST_PATH = "../Project1/SentimentDataset/Test/"
# POS_CORPORA = "../Project1/SentimentDataset/Train/pos"
# NEG_CORPORA = "../Project1/SentimentDataset/Train/neg"

DEV_PATH = "../SentimentAnalysis/datasets/"
TEST_PATH = "../SentimentAnalysis/datasets/test"
POS_CORPORA = "../SentimentAnalysis/datasets/pos"
NEG_CORPORA = "../SentimentAnalysis/datasets/neg"

def unigramSentiment(negTrained, posTrained, sentence):
    posPerplexity = perplexity.perplexity_unigram(sentence, posTrained.unigram)
    negPerplexity = perplexity.perplexity_unigram(sentence, negTrained.unigram)
    if posPerplexity < negPerplexity:
        return 0
    else:
        return 1

def bigramSentiment(negTrained, posTrained, sentence):
    posPerplexity = perplexity.perplexity_bigram(sentence, posTrained.bigram)
    negPerplexity = perplexity.perplexity_bigram(sentence, negTrained.bigram)
    if posPerplexity < negPerplexity:
        return 0
    else:
        return 1

def combinedSentiment(negTrained, posTrained, sentence, weightBi):
    uniPosPP = perplexity.perplexity_unigram(sentence, posTrained.unigram)
    biPosPP = perplexity.perplexity_bigram(sentence, posTrained.bigram)

    uniNegPP = perplexity.perplexity_unigram(sentence, negTrained.unigram)
    biNegPP = perplexity.perplexity_bigram(sentence, negTrained.bigram)

    posPP = weightBi * biPosPP + (1 - weightBi) * uniPosPP
    negPP = weightBi * biNegPP + (1 - weightBi) * uniNegPP

    if posPP < negPP:
        return 0
    else:
        return 1

def classifyTest(filePath, test, model, posModel, negModel, weightBi):
    """ Generates an output text file, where each line is of the form
    <line_id>, <classification>

    line_id        : the line number of the review being classified
    classification : 1 if the review is negative, 0 if it is positive """

    outFilePath = "negoutput" + test + "_" + model + ".txt"
    with open(outFilePath, 'w') as tfile:
        fileHandle = open(filePath, 'r')
        lineCount = 1
        for line in fileHandle:
            line = line.rstrip('\n')
            line = line #add sentence beginning marker
            result = -1
            if model.lower() == "unigram":
                result = unigramSentiment(negModel, posModel, line)
            elif model.lower() == "bigram":
                result = bigramSentiment(negModel, posModel, line)
            elif model.lower() == "combined":
                result = combinedSentiment(negModel, posModel, line, weightBi)
            tfile.write(str(lineCount) + "," + str(result))
            tfile.write("\n")
            lineCount = lineCount + 1

def count(filepath):
    filehandler = open(filepath, 'r')
    count = 0
    for line in filehandler:
        line = line.replace(' ', "")
        line = line.replace('\n', "")
        tokens = line.split(',')
        print(tokens)
        count = count + int(tokens[1])
    print(count)
    return count

if __name__ == '__main__':
    k = float(sys.argv[3])
    posTrain = simple_smoothing.AddKSmoothing(corpus.Corpus(POS_CORPORA), k)
    negTrain = simple_smoothing.AddKSmoothing(corpus.Corpus(NEG_CORPORA), k)

    testName = sys.argv[1]
    model = sys.argv[2]
    weightBi = float(sys.argv[4])
    filePath = DEV_PATH + testName + ".txt"

    classifyTest(filePath, testName, model, posTrain, negTrain,weightBi)
    count("negoutput" + testName + "_" + model + ".txt")

# python classification.py <corpus> <model> <k> <weightBi>
# python classification.py neg bigram 1 0.5
# <UNK> doesn't have to be in the following word FIXED
# change k to float -> tell zach to remove assertion
# how do we know that every word will be followed by UNK
