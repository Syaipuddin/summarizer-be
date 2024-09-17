import re
from datetime import datetime
from math import ceil

import numpy as np
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer
from preprocessor import PreProcess
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
factory = StopWordRemoverFactory()
stopwords = factory.get_stop_words()

from indonesian_tokenizer import IndonesianTokenizer

os.environ['PYTHONIOENCODING'] = 'utf-8'

pp = PreProcess()


def preprocess_docs_for_pipeline(docs_array):
    normalized_docs = []
    for doc in docs_array:
        normalized_docs.append(pp.start_sentence_for_training(doc))

    print("Finished Preprocessing")
    return [i for i in normalized_docs if i != '']

class Summarizer:

    pipeline = False
    lsa_models = False
    preprocess = False

    def __init__(self):

        self.pipeline = Pipeline([
            ('preprocess', FunctionTransformer(func=preprocess_docs_for_pipeline, validate=False)),
            ('tfidf', TfidfVectorizer(stop_words=stopwords)),
            ('svd', TruncatedSVD(n_components=50))
        ])
        self.load_models()

    def load_models(self):
        path = "models"
        first_dir = self.find_models(path)
        if os.path.isfile(f"{path}/{first_dir}.joblib"):
            self.lsa_models = joblib.load(f"{path}/{first_dir}.joblib")

    def find_neighbours(self, sentences, query_setences):

        X_reduced = self.lsa_models.transform(sentences + [query_setences])

        X_reduced_articles = X_reduced[:-1]
        X_reduced_query = X_reduced[-1]

        knn = NearestNeighbors(n_neighbors=3, metric='cosine')
        knn.fit(X_reduced_articles)

        distances, indices = knn.kneighbors([X_reduced_query])

        most_relevant_sentences = []

        for i in indices[0]:
            most_relevant_sentences.append(sentences[i])

        return most_relevant_sentences

    def summarize_sumy(self, input_text):
        # Parse the input text
        parser = PlaintextParser.from_string(input_text, IndonesianTokenizer())

        # Create an LSA summarizer
        summarizer = LsaSummarizer()

        # Generate the summary
        summary = summarizer(parser.document,sentences_count=3)  # You can adjust the number of sentences in the summary

        summed_articles = []
        for sentence in summary:
            summed_articles.append(str(sentence))

        return summed_articles

    def summarize(self, input_text):

        norm_sentences, sentences = pp.start_sentence(input_text)

        if self.lsa_models and norm_sentences:
            print(norm_sentences)
            X_reduced = self.lsa_models.transform(norm_sentences)
            sentence_scores = np.linalg.norm(X_reduced, axis=1)
            ranking = sentence_scores.argsort()[::-1]

            # Choose the top N sentences
            N = 3  # Number of sentences for summary
            top_sentences = [sentences[i] for i in ranking[:N]]

            return top_sentences

        else:
            raise Exception("Models not trained, please train model first")

    def find_models(self, directory):
        try:
            # List all entries in the directory
            entries = os.listdir(directory)
            # Filter out files, keeping only directories
            subfolders = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]
            # Sort subfolders alphabetically
            subfolders.sort(reverse=True)
            # Return the first subfolder if available
            if subfolders:
                return subfolders[0]
            else:
                return None
        except FileNotFoundError:
            print(f"The directory {directory} does not exist.")
            return None