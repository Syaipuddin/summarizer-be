from sumy.nlp.tokenizers import Tokenizer
from nltk import sent_tokenize, word_tokenize
from nlp_id.tokenizer import Tokenizer as IDTokenizer
tokenizer = IDTokenizer()

class IndonesianTokenizer(Tokenizer):
    def __init__(self):
        self.id_tokenizer = IDTokenizer()

    def tokenize(self, text):
        return self.id_tokenizer.tokenize(text)

    def to_sentences(self, text):
        return tuple(sent_tokenize(text))

    def to_words(self, sentence):
        return tuple(word_tokenize(sentence))