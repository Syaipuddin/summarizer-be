import base64
from datetime import datetime

import joblib
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from rouge import Rouge
from flask import Flask, request
from preprocessor import PreProcess
from summarizer import Summarizer
from news_fetcher import NewsFetcher
from flask_cors import CORS
import pandas as pd
import re
import os

pp = PreProcess()
summ = Summarizer()
nf = NewsFetcher()

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

UPLOAD_FOLDER = 'data'  # Change this to your desired path
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def is_url(string):
    # Regular expression pattern for identifying URLs
    url_pattern = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # ...or ipv4
        r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # ...or ipv6
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    return re.match(url_pattern, string) is not None


def separate_urls(data):
    urls = []
    non_urls = []

    for item in data:
        if is_url(item['link']):
            urls.append(item['link'])
        else:
            non_urls.append(item['link'])

    return urls, non_urls


@app.route('/summarize', methods=['POST'])
def summarize_article():
    data = request.get_json()
    body = data['body']
    title = data['title'] if data['title'] else ''

    try:

        urls, docs = separate_urls(body)
        news_data = nf.get_news(urls)

        all_paragraphs = news_data + docs

        summarized_article = []
        for i in all_paragraphs:
            for j in summ.summarize(i):
                summarized_article.append(j)

        nearest_neighbours = summ.find_neighbours(summarized_article, title if title else summarized_article[0])

        return {
            "sum_text": ' '.join(nearest_neighbours)
        }

    except Exception as ex:
        print(ex)
        return { 'sum_text' : f"{ex}"}


@app.route("/train-model", methods=['POST'])
def train_model():
    f = request.files['file']

    from werkzeug.utils import secure_filename
    filename = secure_filename(f.filename)
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    num_of_docs = 10
    data = pd.read_json(f'data/{filename}', lines=True)
    data = data.dropna(axis=1)
    data = data.head(num_of_docs)

    all_articles = []

    print("Fetching articles")
    for i, row in data.iterrows():
        print(row['source_url'])
        article = nf.get_one_news(row['source_url'])
        if article:
            all_articles.append(article)

    print(all_articles)
    print("Training Model")
    pipeline = summ.pipeline.fit(all_articles)
    summ.lsa_models = pipeline

    print("storing model")
    dirname = os.path.dirname(__file__)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(dirname, f"./models/{timestamp}.joblib")
    joblib.dump(summ.lsa_models, filename)

    print("Summarizing Article")
    summarized_articles = []
    for i in all_articles:
        summed = summ.summarize(i)
        summarized_articles.append({
            "doc": i,
            "summed": ' '.join(summed)
        })

    hyps, refs = map(list, zip(*[[d['doc'], d['summed']] for d in summarized_articles]))
    rouge = Rouge()
    scores = rouge.get_scores(hyps, refs, avg=True)

    get_fig = plot_scores(scores, "training_rouge.png")

    response = {
        "data" : summarized_articles,
        "scores" : scores,
        "fig": png_to_base64(get_fig)
    }
    return response

@app.route("/test-model", methods=['POST'])
def test_model():
    f = request.files['file']

    from werkzeug.utils import secure_filename
    filename = secure_filename(f.filename)
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    num_of_docs = 10
    data = pd.read_json(f'data/{filename}', lines=True)
    data = data.dropna(axis=1)
    data = data.head(num_of_docs)

    all_articles = []

    print("Fetching articles")
    for i, row in data.iterrows():
        print(row['source_url'])
        article = nf.get_one_news(row['source_url'])
        if article:
            all_articles.append(article)

    print("Summarizing Article")
    summarized_articles = []
    for i in all_articles:
        summed = summ.summarize(i)
        summarized_articles.append({
            "doc": i,
            "summed": ' '.join(summed)
        })

    hyps, refs = map(list, zip(*[[d['doc'], d['summed']] for d in summarized_articles]))
    rouge = Rouge()
    scores = rouge.get_scores(hyps, refs, avg=True)

    get_fig = plot_scores(scores, "training_rouge.png")

    response = {
        "data" : summarized_articles,
        "scores" : scores,
        "fig": png_to_base64(get_fig)
    }
    return response


def plot_scores(scores, filename):
    # Extract data for plotting
    rouge_metrics = list(scores.keys())
    f_scores = [scores[metric]['f'] for metric in rouge_metrics]
    p_scores = [scores[metric]['p'] for metric in rouge_metrics]
    r_scores = [scores[metric]['r'] for metric in rouge_metrics]

    x = range(len(rouge_metrics))

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.bar(x, f_scores, width=0.2, label='F Score', align='center')
    plt.bar([p + 0.2 for p in x], p_scores, width=0.2, label='Precision', align='center')
    plt.bar([r + 0.4 for r in x], r_scores, width=0.2, label='Recall', align='center')

    plt.xlabel('ROUGE Metrics')
    plt.ylabel('Scores')
    plt.title('ROUGE Scores')
    plt.xticks([p + 0.2 for p in x], rouge_metrics)
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(filename)
    plt.close()

    return filename


def png_to_base64(png_path):
    with open(png_path, "rb") as image_file:
        # Read the image as bytes
        image_bytes = image_file.read()
        # Encode the bytes as Base64
        base64_encoded = base64.b64encode(image_bytes).decode('utf-8')
        return base64_encoded





