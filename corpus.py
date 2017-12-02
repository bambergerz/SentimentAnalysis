from corpus_tools.corpus_entry import Entry

START_TOKEN = "<s>"
UNKNOWN_TOKEN = "<UNK>"
NON_TOKENS = ["'", "`", "''", "``"]
END_TOKENS = [".", "?", "!"]

class Corpus():
    def __init__(self, name, to_parse=True):
        """
        
        :param name: the name of the file containing the corpus without the ".txt" suffix. 
        """

        self.name = name
        #dictionary of token -> corpus_entry instance
        if to_parse:
            self._corpus_entry_dict = self.parser(name + ".txt")
        else:
            self._corpus_entry_dict = {}

    def __repr__(self):
        result = "< Corpus object at " + str(hex(id(self))) + ">\n"
        return result

    @property
    def entries(self):
        """
        
        :return: the corpus entry dictionary of this Corpus object
        """
        return self._corpus_entry_dict

    def set_corpus_entry_dict(self, dict):
        """
        
        :param dict: the corpus entry dict to which we will set the corpus entry dict attribute. 
        :return: Nothing
        """
        self._corpus_entry_dict = dict

    def compare_entry_dicts(self, other):
        """
        
        :param other: another corpus
        :return: a string showing the values in self vs. other
        """
        result = ""
        for key, value in self._corpus_entry_dict.items():
            data = str(key) + ": " + str(value.frequency) + "," +\
                   str(other.entries[key].frequency) + "\n"
            result += data
            for x, y in value.following_words.items():
                result += "    " + str(x) + ":" + str(y) + "," +\
                          str(other.entries[key].following_words[x]) + "\n"
        return result

    def print_corpus_entry_frequencies(self):
        """
        
        :return: A string representation of the entries of this array and their frequencies
        """
        result = ""
        for key, value in self._corpus_entry_dict.items():
            data = str(key) + " : " + str(value.frequency) + "\n"
            result += data
        return result

    def parser(self, filepath):
        """
        :param filepath: the name of the file containing our corpus which we intend to parse. 
        :return: a dictionary of corpus entries. Eventually becomes the '_corpus_entry_dict' attribute. """
        
        seenWords = set()
        result = {}
        
        #Start Token Added to Corpus
        start_token = Entry(START_TOKEN)
        #start_token.increment_frequency()
        result[START_TOKEN] = start_token
        
        #Unknown Token Added to Corpus
        unknown_token = Entry(UNKNOWN_TOKEN)
        result[UNKNOWN_TOKEN] = unknown_token
        
        fileHandle = open(filepath, 'r')
        for line in fileHandle:
            line = line.rstrip('\n')
            tokens = line.split(' ')
            prev_word = START_TOKEN
            result[START_TOKEN].increment_frequency()
            for token in tokens:
                #if the token has never been seen, tag it as UNK
                if token not in seenWords:
                    seenWords.add(token)
                    token = UNKNOWN_TOKEN
                    
                if token in result.keys(): 
                    result[token].increment_frequency()
                else:
                    new_token = Entry(token)
                    new_token.increment_frequency()
                    result[token] = new_token
                result[prev_word].add_following_word(token)
                prev_word = token
        return result
