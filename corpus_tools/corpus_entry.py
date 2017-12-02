import collections


class Entry(object):
    def __init__(self, token):
        """
        
        :param token: A token entry in a corpus. This is a string 
        """
        self._id = token
        self._frequency = 0
        self._following_words = collections.OrderedDict()

    @property
    def id(self):
        """
        
        :return: the string representing which token this is in the corpus
        """
        return self._id

    @property
    def frequency(self):
        """
        
        :return: the frequency with which this token appears in the corpus
        """
        return self._frequency

    @property
    def following_words(self):
        """
        
        :return: The dictionary of following words and their frequency
        """
        return self._following_words

    def add_following_word(self, word):
        keys = self._following_words.keys()
        if word in keys:
            self._following_words[word] += 1
        else:
            self._following_words[word] = 1
        return True

    def get_following_word_frequency(self, word):
        """
        
        :param word: The word we are determining the frequency with which it follows the current token.
        :return: the frequency with which 'word' follows the current token.
        An integer. If 'word' does not follow the current token, return 0
        """
        if word in self._following_words.keys():
            return self._following_words[word]
        else:
            return 0

    def increment_frequency(self):
        """
        
        :return: incrememnt the frequency with which this token appears in the corpus. 
        """
        self._frequency += 1