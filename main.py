import re

from flask_cors import CORS
from flask import Flask, request
from preprocessor import PreProcess
from summarizer import Summarizer
from news_fetcher import NewsFetcher
from flask_cors import CORS, cross_origin
import numpy as np

pp = PreProcess()
summ = Summarizer()
nf = NewsFetcher()

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/summarize', methods=['POST'])
def summarize_article():
    data = request.get_json()
    urls = data['urls']
    all_paragraphs = nf.get_news(urls)

    try:
        summed_article = []
        for sent in all_paragraphs:
            if sent:
                norm_sent, sentence = pp.start_sentence(sent)
                summed_article.append(summ.summarize(sentence, norm_sent))

        knn_summed_article = summ.summarize_knn(summed_article)

        cleaned_knn_summed = []
        for i in knn_summed_article:
            # Remove Unicode characters that are not displaying properly
            cleaned_sentence = re.sub(r'[^\x00-\x7F]+', '', i)
            # Remove newline characters and replace with spaces
            cleaned_sentence = cleaned_sentence.replace('\n', ' ').replace('\r', ' ')
            cleaned_knn_summed.append(cleaned_sentence)

        text = ' '.join(cleaned_knn_summed)
        text = text.replace('\n', ' ')

        norm_text, text = pp.start_sentence(text)
        summed_text = summ.summarize(text, norm_text)
        # summ

        norm_summ, summed_text = pp.start_sentence(summed_text)

        response = {
            'sum_text': '. '.join(list(set(norm_summ)))
        }

        return response

    except:
        return { 'sum_text' : "Gagal mengambil berita"}





