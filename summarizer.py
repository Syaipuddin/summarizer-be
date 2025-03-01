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
            ('tfidf', TfidfVectorizer(stop_words=stopwords, ngram_range=(1, 2), max_features=5000)),
            ('svd', TruncatedSVD(n_components=50, n_iter=25))
        ])
        # self.load_models()

    def load_models(self):
        path = "models"
        model_file = self.find_models(path)
        if model_file is not None and os.path.isfile(os.path.join(path, model_file)):
            self.lsa_models = joblib.load(os.path.join(path, model_file))
            print(f"Model '{model_file}' loaded successfully.")
        else:
            raise Exception("no job found, please train model first")

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
            N = 4  # Number of sentences for summary
            top_sentences = [sentences[i] for i in ranking[:N]]

            return top_sentences

        else:
            raise Exception("Models not trained, please train model first")

    def summarize_adjust(self, input_text):

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
            joblib_files = [entry for entry in entries if entry.endswith('.joblib')]
            # Sort subfolders alphabetically
            joblib_files.sort(reverse=True)
            # Return the first subfolder if available
            if joblib_files:
                return joblib_files[0]
            else:
                return None
        except FileNotFoundError:
            print(f"The directory {directory} does not exist.")
            return None

    def filter_low_feature_docs(self, docs, min_features=50):
        filtered_docs = []
        for doc in docs:
            processed_doc = pp.start_sentence_for_training(doc)
            vectorizer = TfidfVectorizer(stop_words=stopwords)
            tfidf_vector = vectorizer.fit_transform([processed_doc])

            if len(vectorizer.get_feature_names_out()) >= min_features:
                filtered_docs.append(doc)

            print(len(vectorizer.get_feature_names_out()))

        return filtered_docs

    def preprocess_docs_for_pipeline(self, docs_array):
        normalized_docs = []
        for doc in docs_array:
            normalized_docs.append(pp.start_sentence_for_training(doc))

        print("Finished Preprocessing")
        return [i for i in normalized_docs if i != '']