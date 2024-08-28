from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
import numpy as np
import nltk
import re
import string

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')


class PreProcess:
    def start_words(self, msg):
        normalized_sentences = self.normalize(msg)
        tokenize_words = self.word_tokenization(normalized_sentences)
        text = ' '.join(tokenize_words)

        return text, msg

    def start_sentence(self, msg):

        sentences = self.sent_tokenization(msg)
        normalize_corpus = np.vectorize(self.normalize)
        normalized_sentences = normalize_corpus(sentences)

        return normalized_sentences, sentences
    
    def normalize(self, msg):
        
        cf = self.case_fold(msg)
        nn = self.no_noise(cf)
        ns = self.no_stopwords(nn)
        text = ''.join(ns)

        return text
    
    # CASE FOLDING
    def case_fold(self, msg):
        case_folded = msg.lower()
        numberless = re.sub(r"\d+", "", case_folded)
        no_punc = numberless.translate(str.maketrans("", "", string.punctuation))
        no_ws = no_punc.strip()

        return no_ws

    # STOP WORDS REMOVAL
    def no_stopwords(self, msg_list):
        stopwords_list = set(stopwords.words('indonesian'))
        cleaned = []
        for words in msg_list:
            if words not in stopwords_list:
                cleaned.append(words)

        return cleaned

        # STEMMING

    def stemmer(self, msg):
        stemFact = StemmerFactory()
        stemmer = stemFact.create_stemmer()
        stemmed_words = stemmer.stem(msg)

        return stemmed_words

    # NOISE REMOVAL
    def no_noise(self, msg):
        clean_text = re.sub(r'[\.\?\!\,\:\;\"]', '', msg)
        return clean_text

    def sent_tokenization(self, msg):
        tokenized = sent_tokenize(msg)
        return tokenized

    def word_tokenization(self, msg):
        tokenized = word_tokenize(msg)
        return tokenized