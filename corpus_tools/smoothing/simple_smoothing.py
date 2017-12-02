from corpus_tools.corpus import Corpus
from corpus_tools.corpus_entry import Entry
from corpus_tools.smoothing.enums.smoothing_methods import Smoothing_Methods
import numpy as np

class Smoothing(object):
    def __init__(self, method, corpus):
        """
        
        :param method: the smoothing method we will be utilizing. A string. 
        :param corpus: a corpus object which forms the basis for the smoothing. 
        """
        self.assert_corpus(corpus)
        self._method = method
        self._corpus = corpus

    def __repr__(self):
        """
        
        :return: a string representation of a generic Smoothing object
        """
        return "< " + str(self._method) + " smoothing object at " + str(hex(id(self))) + ">"

    @property
    def method(self):
        """
        
        :return: the method used for this smoothing
        """
        return self._method

    @property
    def corpus(self):
        """
        
        :return: the corpus of this object
        """
        return self._corpus

    #######################################
    ########## ASSERTION HELPERS ##########
    #######################################

    def assert_corpus(self, corpus):
        """
        
        :param corpus: the object whose type we are asserting is a corpus
        :return: true if corpus is in fact a Corpus. False otherwise. 
        """
        assert isinstance(corpus, Corpus), "Entered value is not a Corpus object"

    def assert_num(self, num):
        """
        
        :param num: the object whose type we are asserting is an int
        :return: true if num is in fact an int. False otherwise. 
        """
        assert isinstance(num, int), "Entered value is not an int"

    #######################################
    ############# INIT HELPERS ############
    #######################################

    def get_corpus_size(self):
        """
        
        :return: return the size of the corpus (i.e., the sum of the frequencies of all words). 
        """
        self.assert_corpus(self.corpus)
        result = 0
        for key, value in self._corpus.entries.items():
            result += self.corpus.entries[key].frequency
        return result


class AddKSmoothing(Smoothing):
    def __init__(self, corpus, k):
        """
        
        :param method: our desired smoothing method
        :param corpus: the corpus we are using for smoothing
        """
        super(AddKSmoothing, self).assert_corpus(corpus)
        #assert isinstance(k, int), "k must be an integer"
        self._k = k
        if k == 1:
            super(AddKSmoothing, self).__init__(Smoothing_Methods.ADDONE, corpus)
        else:
            super(AddKSmoothing, self).__init__(Smoothing_Methods.ADDK, corpus)
        self._N = self.get_corpus_size()
        self._V = len(self.corpus.entries)

    def unigram(self, word):
        """
        
        :param word: the word we are determining the probability of. A string
        :return: the AddOne probability of word W_x to appear. A float
        """
        if word in self.corpus.entries.keys():                                                      # known unigram
            num = (self.corpus.entries[word].frequency + self._k)                                   #
            dem = (self._N + (self._k * self._V))                                                   #
        else:                                                                                       # unkown unigram
            num = self.corpus.entries["<UNK>"].frequency + self._k                                  #
            dem = (self._N +(self._k * self._V))                                                    #
        return float(num) / dem                                                                            #

    def bigram(self, word, prev_word):
        """
        
        :param word: The word we are trying to compute the probability of given 
        a bigram model and an add-one smoothing of the corpus. A string.
        Appears in the corpus.
        :param prev_word: The previous word which appeared in the sequence. 
        A string. Appears in the corpus. 
        :return: The AddOne probability of the word W_x to appear. A float.
        """
        if word in self.corpus.entries.keys():                                                      #
            if prev_word in self.corpus.entries.keys():                                             #
                if word in self.corpus.entries[prev_word].following_words.keys():                   # KNOWN BIGRAM
                    num = (self.corpus.entries[prev_word].following_words[word] + self._k)          #
                    den = (self.corpus.entries[prev_word].frequency + self._k * self._V)            #
                else:                                                                               # UNKOWN BIGRAM
                    num = self._k                                                                   #
                    den = (self.corpus.entries[prev_word].frequency + self._k * self._V)            #
            else:                                                                                   # UNKOWN KNOWN
                if word in self.corpus.entries["<UNK>"].following_words.keys():                     #
                    num = self.corpus.entries["<UNK>"].following_words[word] + self._k              #
                else:                                                                               #
                    num = self._k                                                                   # keyerror pos.
                    den = self.corpus.entries["<UNK>"].frequency + self._k * self._V                #
        else:                                                                                       #
            if prev_word in self.corpus.entries.keys():                                             # KNOWN UNKOWN
                if "<UNK>" in self.corpus.entries[prev_word].following_words.keys():                #
                    num = self.corpus.entries[prev_word].following_words["<UNK>"] + self._k         #
                else:                                                                               #
                    num = self._k                                                                   # keyerror
                den = self.corpus.entries[prev_word].frequency + self._k * self._V                  #
            else:                                                                                   # UNKOWN UNKOWN
                num = self.corpus.entries["<UNK>"].following_words["<UNK>"] + self._k               #
                den = self.corpus.entries["<UNK>"].frequency + self._k * self._V                    #
        return num / float(den)                                                                     #