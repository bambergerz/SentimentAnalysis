import sys
import os
from corpus_tools import corpus
import numpy
import random

path = os.getcwd()
POS_CORPORA = os.path.join(path, "datasets", "pos")
NEG_CORPORA = os.path.join(path, "datasets", "neg")
POS_UNLABELED = os.path.join(path, "datasets", "pos_unlabeled")
NEG_UNLABELED = os.path.join(path, "datasets", "neg_unlabeled")
POS_TRAIN = os.path.join(path, "datasets", "pos_train")
NEG_TRAIN = os.path.join(path, "datasets", "neg_train")
POS_TEST = os.path.join(path, "datasets", "pos_test")
NEG_TEST = os.path.join(path, "datasets", "neg_test")
POS_TRAIN_TEMP = POS_TRAIN + "_temp"
NEG_TRAIN_TEMP = NEG_TRAIN + "_temp"

POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"


class NaiveBayes:
    def __init__(self, poscorp, negcorp):
        """

        :param poscorp: The positive corpus. Contains postivie reviews.
        :param negcorp: The negative corpus. Contains negative reviews.
        """
        self.vocabsize = len(poscorp.entries) + len(negcorp.entries)
        self.poscorp = poscorp
        self.negcord = negcorp

        # Initialize priors
        self.num_pos = poscorp.lines
        self.num_neg = negcorp.lines
        self.num_total = self.num_pos + self.num_neg
        self.pos_prior = self.num_pos / self.num_total
        self.neg_prior = self.num_neg / self.num_total

        # Initialize likelyhoods
        self.pos_likelyhoods = {}
        self.neg_likelyhoods = {}

        for key, value in poscorp.entries.items():
            self.pos_likelyhoods[key] = (poscorp.entries[key].frequency + 1) /\
                                        (self.vocabsize + poscorp.num_tokens)

        for key, value in negcorp.entries.items():
            self.neg_likelyhoods[key] = (negcorp.entries[key].frequency + 1) /\
                                        (self.vocabsize + negcorp.num_tokens)

    def eval(self, review):
        """

        :param review: A sentence representing a movie review
        :return: POSITIVE if the movie is deemed to have a positive sentiment. NEGATIVE otherwise.
        """
        pos_prod = 1
        neg_prod = 1
        for word in review.split():
            # print("    word: " + word)
            try:
                # print("    pos_likelyhoods[word] = " + str(self.pos_likelyhoods[word]))
                pos_prod *= (self.pos_likelyhoods[word])
            except KeyError as e:
                # print("    pos likelyhoods[word] = 0")
                pos_prod *= (1 / (self.vocabsize + self.poscorp.num_tokens))
            try:
                # print("    neg likeyhoods[word] = " + str(self.neg_likelyhoods[word]))
                neg_prod *= (self.neg_likelyhoods[word])
            except KeyError as e:
                # print("    neg likelyhoods[word] = 0")
                neg_prod *= (1 / (self.vocabsize + self.negcord.num_tokens))
            # print("    pos prod: " + str(pos_prod))
            # print("    neg prod: " + str(neg_prod))

        pos_likelyhood = self.pos_prior * pos_prod
        neg_likelyhood = self.neg_prior * neg_prod
        # print("pos_likelyhood: " + str(pos_likelyhood))
        # print("neg likelyhood: " + str(neg_likelyhood))

        if pos_likelyhood > neg_likelyhood:
            return POSITIVE
        else:
            return NEGATIVE


def partition():
    """

    :return: None, but partition the positive and negative corpuses into training, unlabeled, and test datasets.
    """

    pos_unlabeled_lines = []
    pos_training_lines = []
    pos_test_lines = []
    pos_lines = []
    with open(POS_CORPORA + ".txt", 'r') as fileHandle:
        pos_lines += fileHandle.readlines()

    for line in pos_lines:
        if random.randint(0, 1):
            pos_unlabeled_lines.append(line)
        else:
            if random.randint(0,4) > 1:
                pos_training_lines.append(line)
            else:
                pos_test_lines.append(line)

    neg_unlabeled_lines = []
    neg_training_lines = []
    neg_test_lines = []
    neg_lines = []

    with open(NEG_CORPORA + ".txt", 'r') as fileHandle:
        neg_lines += fileHandle.readlines()

    for line in neg_lines:
        if random.randint(0,1):
            neg_unlabeled_lines.append(line)
        else:
            if random.randint(0,4) > 1:
                neg_training_lines.append(line)
            else:
                neg_test_lines.append(line)

    with open(POS_UNLABELED + ".txt", 'w') as fileHandle:
        fileHandle.writelines(pos_unlabeled_lines)
    with open(POS_TRAIN + ".txt", 'w') as fileHandle:
        fileHandle.writelines(pos_training_lines)
    with open(POS_TEST + ".txt", 'w') as fileHandle:
        fileHandle.writelines(pos_test_lines)

    with open(NEG_UNLABELED + ".txt", 'w') as fileHandle:
        fileHandle.writelines(neg_unlabeled_lines)
    with open(NEG_TRAIN + ".txt", 'w') as fileHandle:
        fileHandle.writelines(neg_training_lines)
    with open(NEG_TEST + ".txt", 'w') as fileHandle:
        fileHandle.writelines(neg_test_lines)


if __name__ == "__main__":

    pos_corpus = corpus.Corpus(POS_CORPORA)
    neg_corpus = corpus.Corpus(NEG_CORPORA)

    partition()
    results = []

    for _ in range(5):

        nb_models = []

        for x in range(25):
            pos_training_sentences = []
            neg_training_sentences = []

            with open(POS_TRAIN + ".txt", 'r') as fileHandle:
                pos_training_sentences += fileHandle.readlines()
            with open(NEG_TRAIN + ".txt", 'r') as fileHandle:
                neg_training_sentences += fileHandle.readlines()

            # choose sentences with replacement
            pos_training_sentences = numpy.random.choice(pos_training_sentences, len(pos_training_sentences), True)
            neg_training_sentences = numpy.random.choice(neg_training_sentences, len(neg_training_sentences), True)

            # write these chosen lines into a file
            with open(POS_TRAIN_TEMP + ".txt", 'w') as fileHandle:
                fileHandle.writelines(pos_training_sentences)
            with open(NEG_TRAIN_TEMP + ".txt", 'w') as fileHandle:
                fileHandle.writelines(neg_training_sentences)

            # build corpuses
            pos = corpus.Corpus(POS_TRAIN_TEMP)
            neg = corpus.Corpus(NEG_TRAIN_TEMP)

            # train NB model
            nb_models.append(NaiveBayes(pos, neg))

        # Combine positive and negative unlabeled sentences. Shuffle resulting list.
        unlabeled_sentences = []
        with open(POS_UNLABELED + ".txt", 'r') as fileHandle:
            unlabeled_sentences += fileHandle.readlines()
        with open(NEG_UNLABELED + ".txt", 'r') as fileHandle:
            unlabeled_sentences += fileHandle.readlines()
        random.shuffle(unlabeled_sentences)

        # Add sentences which all models agree on to the training data
        add_to_training = []
        for sentence in unlabeled_sentences:
            classifications = set()
            for model in nb_models:
                classifications.add((sentence, model.eval(sentence)))
            if len(classifications) == 1:
                add_to_training.append(classifications.pop())

        pos_training_sentences = []
        neg_training_sentences = []
        with open(POS_TRAIN + ".txt", 'r') as fileHandle:
            pos_training_sentences += fileHandle.readlines()
        with open(NEG_TRAIN + ".txt", 'r') as fileHandle:
            neg_training_sentences += fileHandle.readlines()
        for line in add_to_training:
            if line[1] == POSITIVE:
                pos_training_sentences.append(line[0])
            else:
                neg_training_sentences.append(line[0])
        with open(POS_TRAIN + ".txt", 'w') as fileHandle:
            fileHandle.writelines(pos_training_sentences)
        with open(NEG_TRAIN + ".txt", 'w') as fileHandle:
            fileHandle.writelines(neg_training_sentences)

        # gather PRF scores:
        pos_test_sentences = []
        neg_test_sentences = []
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        with open(POS_TEST + ".txt", 'r') as fileHandle:
            pos_test_sentences += fileHandle.readlines()
        with open(NEG_TEST + ".txt", 'r') as fileHandle:
            neg_test_sentences += fileHandle.readlines()

        for sentence in pos_test_sentences:
            pos_count = 0
            neg_count = 0
            for model in nb_models:
                if model.eval(sentence) == POSITIVE:
                    pos_count += 1
                else:
                    neg_count += 1
            if pos_count > neg_count:
                true_positives += 1
            else:
                false_negatives += 1

        for sentence in neg_test_sentences:
            pos_count = 0
            neg_count = 0
            for model in nb_models:
                if model.eval(sentence) == POSITIVE:
                    pos_count += 1
                else:
                    neg_count += 1
            if pos_count > neg_count:
                false_positives += 1
            else:
                true_negatives += 1

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        fmeasure = (2 * precision * recall) / (precision + recall)

        results.append((precision, recall, fmeasure))

    for prf in results:
        print(prf)



    # nb = NaiveBayes(pos_corpus, neg_corpus)

    # reviews = ["good movie",
    #            "not good",
    #            "I thought it was average",
    #            "great actors",
    #            "good",
    #            "I liked it !",
    #            "the movie was intriguing"]
    # print("Naive Baye's details:\n\npositive prior: " + str(nb.pos_prior) +
    #       "\nnegative prior: " + str(nb.neg_prior) + "\nvocab size: " + str(nb.vocabsize) + "\n")
    # for review in reviews:
    #     print("\"" + review + "\"" + " -- " + str(nb.eval(review)) + "\n" * 2)
    #
    # while(1):
    #     review = input("Please write a sample movie review:\n")
    #     print("\"" + review + "\"" + " -- " + str(nb.eval(review)) + "\n" * 2)
