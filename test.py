from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np
from preprocessor import PreProcess
from summarizer import Summarizer

# Step 1: Generate the synthetic dataset (paragraphs)
synthetic_paragraphs = [
    "PDIP dan Anies Belum Capai Kesepakatan Politik. Hingga saat ini, belum ada kesepakatan politik yang tercapai antara PDIP dan Anies. Pertemuan yang terjadi hanyalah silaturahim tanpa pembahasan politik apapun karena belum ada posisi untuk membuat kesepakatan. 'Semuanya masih dalam proses,' katanya. PDIP sedang memfinalisasi calon-calon yang akan maju di Pilgub Jakarta. 'Ini masih tahap awal, mari kita tunggu seperti proses di DKI dan Jawa Timur,' jelasnya.",

    "Tidak Ada Kesepakatan Politik Antara PDIP dan Anies. PDIP dan Anies belum mencapai kesepakatan politik apapun. Pertemuan ini murni merupakan bentuk silaturahim tanpa ada kesepakatan politik yang dibuat, karena belum ada keputusan apapun yang bisa diambil. 'Semuanya masih dalam proses,' ucapnya. Saat ini, PDIP sedang menyelesaikan tahap finalisasi calon yang akan maju di Pilgub Jakarta. 'Karena ini baru awal, kita tunggu saja, seperti proses di DKI dan Jawa Timur,' terangnya.",

    "Kesepakatan Politik PDIP dan Anies Masih Dalam Proses. PDIP dan Anies belum mencapai kesepakatan politik apapun hingga saat ini. Pertemuan yang dilakukan hanya sebatas silaturahim tanpa adanya kesepakatan politik, karena belum ada dasar untuk membuat kesepakatan. 'Semuanya masih dalam proses,' katanya. PDIP sedang dalam tahap finalisasi untuk menentukan calon yang akan diusung di Pilgub Jakarta. 'Ini masih awal, mari kita tunggu seperti proses di DKI dan Jawa Timur,' ungkapnya.",

    "Proses Kesepakatan Politik PDIP dan Anies Masih Berjalan. Belum ada kesepakatan politik yang dibuat antara PDIP dan Anies. Pertemuan mereka hanyalah bentuk silaturahim tanpa adanya pembahasan politik, karena belum ada keputusan untuk membuat kesepakatan. 'Semuanya masih dalam proses,' ujarnya. PDIP saat ini sedang melakukan finalisasi calon yang akan maju di Pilgub Jakarta. 'Karena ini masih tahap awal, kita tunggu saja, seperti proses di DKI dan Jawa Timur,' jelasnya.",

    "Silaturahim Tanpa Kesepakatan Politik Antara PDIP dan Anies. Pertemuan antara PDIP dan Anies belum menghasilkan kesepakatan politik. Silaturahim ini tidak membahas kesepakatan politik apapun karena belum ada posisi yang jelas untuk membuat kesepakatan. 'Semuanya masih dalam proses,' kata dia. PDIP sedang dalam proses finalisasi calon-calon yang akan diusung di Pilgub Jakarta. 'Ini masih awal, kita tunggu saja seperti proses di DKI dan Jawa Timur,' tandasnya."
]

# Step 2: Vectorize the paragraphs using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(synthetic_paragraphs)

# Step 3: Fit the k-NN model
k = 1  # Choose the number of neighbors
knn = NearestNeighbors(n_neighbors=k, metric='cosine').fit(X)

# Step 4: Find nearest neighbors for each paragraph
distances, indices = knn.kneighbors(X)

# Step 5: Combine similar paragraphs into one
combined_paragraphs = []
visited = set()

for i, neighbors in enumerate(indices):
    if i not in visited:
        group = [synthetic_paragraphs[j] for j in neighbors if j not in visited]
        combined_paragraph = " ".join(group)
        combined_paragraphs.append(combined_paragraph)
        visited.update(neighbors)

pp = PreProcess()
summ = Summarizer()

normalize_corpus = np.vectorize(pp.start_sentence)
summed_all = []

# Step 6: Print combined paragraphs
for i, paragraph in enumerate(combined_paragraphs, 1):
    norm_sentences, sentences = normalize_corpus(paragraph)
    summed_text = summ.summarize(sentences, norm_sentences)

    summed_all.append(summed_text)

text = ' '.join(summed_all)
text = text.replace('\n', ' ')

norm_text, text = normalize_corpus(text)
summed_text = summ.summarize(norm_text, text)
# summ
print(summed_text)


