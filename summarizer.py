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
        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(vector)

        # find nearest neaghbor for each paragraph
        distances, indices = knn.kneighbors(vector)

        return distances, indices

    def summarize_knn(self, norm_paragraphs, paragraphs):

        vector = self.vectorize(norm_paragraphs)
        distances, indices = self.find_neares_neighbors(vector)

        summs = []
        for i in range(len(paragraphs)):
            sim_idx = indices[i][0]
            summs.append(paragraphs[sim_idx])

        return summs

    def getMatrix(self, sentences):
        tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
        dt_matrix = tv.fit_transform(sentences)
        dt_matrix = dt_matrix.toarray()
        # vocab = dt_matrix.get_feature_names_out()
        td_matrix = dt_matrix.T

        return td_matrix

    def summarize(self, sentences, normalized_sentences, singular_count=2):
        total_sentences = round(len(sentences)*0.25)
        print(total_sentences)

        matrix = self.getMatrix(normalized_sentences)
        u, s, vt = svds(matrix, k=singular_count)

        term_topic_mat, singular_values, topic_document_mat =u, s, vt
        sv_threshold = 0.5
        min_sigma_value = max(singular_values * sv_threshold)
        singular_values[singular_values < min_sigma_value] = 0

        salience_scores = np.sqrt(np.dot(np.square(singular_values), np.square(topic_document_mat)))
        top_sentences_indices = (-salience_scores).argsort()[:total_sentences]
        top_sentences_indices.sort()
        print(top_sentences_indices)
        return '\n' .join(np.array(sentences)[top_sentences_indices])