import sys
import os
from corpus_tools import corpus

path = os.getcwd()
POS_CORPORA = os.path.join(path, "datasets", "pos")
NEG_CORPORA = os.path.join(path, "datasets", "neg")

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
            print("    word: " + word)
            try:
                print("    pos_likelyhoods[word] = " + str(self.pos_likelyhoods[word]))
                pos_prod *= (self.pos_likelyhoods[word])
            except KeyError as e:
                print("    pos likelyhoods[word] = 0")
                pos_prod *= (1 / (self.vocabsize + self.poscorp.num_tokens))
            try:
                print("    neg likeyhoods[word] = " + str(self.neg_likelyhoods[word]))
                neg_prod *= (self.neg_likelyhoods[word])
            except KeyError as e:
                print("    neg likelyhoods[word] = 0")
                neg_prod *= (1 / (self.vocabsize + self.negcord.num_tokens))
            print("    pos prod: " + str(pos_prod))
            print("    neg prod: " + str(neg_prod))

        pos_likelyhood = self.pos_prior * pos_prod
        neg_likelyhood = self.neg_prior * neg_prod
        print("pos_likelyhood: " + str(pos_likelyhood))
        print("neg likelyhood: " + str(neg_likelyhood))

        if pos_likelyhood > neg_likelyhood:
            return POSITIVE
        else:
            return NEGATIVE


if __name__ == "__main__":

    pos_corpus = corpus.Corpus(POS_CORPORA)
    neg_corpus = corpus.Corpus(NEG_CORPORA)
    nb = NaiveBayes(pos_corpus, neg_corpus)
    reviews = ["good movie",
               "not good",
               "I thought it was average",
               "great actors",
               "good",
               "I liked it !"]
    print("Naive Baye's details:\n\npositive prior: " + str(nb.pos_prior) +
          "\nnegative prior: " + str(nb.neg_prior) + "\nvocab size: " + str(nb.vocabsize) + "\n")
    for review in reviews:
        print("\"" + review + "\"" + " -- " + str(nb.eval(review)) + "\n" * 2)
