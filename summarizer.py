from math import floor, ceil

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from scipy.sparse.linalg import svds


class Summarizer:

    def vectorize(self, paragraphs):
        tv = TfidfVectorizer()
        x = tv.fit_transform(paragraphs)

        return x

    def find_neares_neighbors(self, vector):
        # init knn model

        k = 1  # Choose the number of neighbors
        knn = NearestNeighbors(n_neighbors=k, metric='cosine').fit(vector)
        knn.fit(vector)

        # find nearest neaghbor for each paragraph
        distances, indices = knn.kneighbors(vector)

        return distances, indices

    def summarize_knn(self, paragraphs):

        vector = self.vectorize(paragraphs)
        distances, indices = self.find_neares_neighbors(vector)

        combined_paragraphs = []
        visited = set()

        for i, neighbors in enumerate(indices):
            if i not in visited:
                group = [paragraphs[j] for j in neighbors if j not in visited]
                combined_paragraph = " ".join(group)
                combined_paragraphs.append(combined_paragraph)
                visited.update(neighbors)

        # Filter out duplicates using the distances array
        unique_paragraphs = []
        for paragraph in combined_paragraphs:
            if paragraph not in unique_paragraphs:
                unique_paragraphs.append(paragraph)

        return unique_paragraphs

    def getMatrix(self, sentences):
        tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
        dt_matrix = tv.fit_transform(sentences)
        dt_matrix = dt_matrix.toarray()
        # vocab = dt_matrix.get_feature_names_out()
        td_matrix = dt_matrix.T

        return td_matrix

    def summarize(self, sentences, normalized_sentences, singular_count=1):
        total_sentences = floor((len(sentences) * 0.3))
        print(total_sentences)

        matrix = self.getMatrix(normalized_sentences)
        u, s, vt = svds(matrix, k=ceil(total_sentences * 0.6))

        term_topic_mat, singular_values, topic_document_mat =u, s, vt
        sv_threshold = 0.50
        min_sigma_value = max(singular_values * sv_threshold)
        singular_values[singular_values < min_sigma_value] = 0

        salience_scores = np.sqrt(np.dot(np.square(singular_values), np.square(topic_document_mat)))
        top_sentences_indices = (-salience_scores).argsort()[:total_sentences]
        top_sentences_indices.sort()
        # print(top_sentences_indices)

        return '\n' .join(np.array(sentences)[top_sentences_indices])