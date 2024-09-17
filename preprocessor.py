from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
import re
import string

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')


class PreProcess:

    def start_sentence(self, msg):
        if msg:

            norm_sentences = []
            sentences = self.sent_tokenization(msg)
            for i in sentences:
                norm_sentences.append(self.normalize(i))

            filtered_sentences = [sentence for sentence in norm_sentences if sentence != '']
            return filtered_sentences, sentences

    def start_sentence_for_training(self, msg):
        if msg:
            norm_sentences = []
            sentences = self.sent_tokenization(msg)
            for i in sentences:
                norm_sentences.append(self.normalize(i))

            filtered_sentences = [sentence for sentence in norm_sentences if sentence != '']
            return ' '.join(filtered_sentences)
    
    def normalize(self, msg):

        normalized_words = []

        # tokenize to words first
        words = self.word_tokenization(msg)
        for i in words:
            if i != '' or not i.isspace():
                cf = self.case_fold(i)
                nn = self.no_noise(cf)
                ns = self.no_stopwords(nn)
                if ns:
                    normalized_words.append(ns)

        return ' '.join(normalized_words)
    
    # CASE FOLDING
    def case_fold(self, msg):
        case_folded = msg.lower()
        numberless = re.sub(r"\d+", "", case_folded)
        no_punc = numberless.translate(str.maketrans("", "", string.punctuation))
        no_ws = no_punc.strip()

        return no_ws

    # STOP WORDS REMOVAL
    def no_stopwords(self, words):
        stopwords_list = set(stopwords.words('indonesian'))
        if words not in stopwords_list:
            return words

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