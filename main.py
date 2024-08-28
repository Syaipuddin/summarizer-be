from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from preprocessor import PreProcess
from summarizer import Summarizer
import numpy as np

req = Request(
    url="https://www.cnbcindonesia.com/news/20240827162346-4-566791/pdip-beri-kabar-terbaru-soal-anies-rano-karno-simak",
    headers={'User-Agent': 'Mozilla/5.0'}
)

html = urlopen(req)
contents = html.read()
paragraph = ""


soup = BeautifulSoup(contents, 'html.parser')

for script in soup(["script", "style", "header", "footer", "nav", "aside"]):
    script.decompose()

for a in soup.find_all('a', href=True):
    a.extract()

for a in soup.find_all('p', class_='baca'):
    a.extract()

for data in soup.find_all('p'):
    paragraph += data.text

pp = PreProcess()
summ = Summarizer()

# normalize_corpus = np.vectorize(pp.start_sentence)
# norm_sentences, sentences = normalize_corpus(paragraph)
# summed_text = summ.summarize(sentences, norm_sentences)
# print(summed_text)

paragraphs = [
    "PDIP dan Anies Belum Capai Kesepakatan Politik. Hingga saat ini, belum ada kesepakatan politik yang tercapai antara PDIP dan Anies. Pertemuan yang terjadi hanyalah silaturahim tanpa pembahasan politik apapun karena belum ada posisi untuk membuat kesepakatan. 'Semuanya masih dalam proses,' katanya. PDIP sedang memfinalisasi calon-calon yang akan maju di Pilgub Jakarta. 'Ini masih tahap awal, mari kita tunggu seperti proses di DKI dan Jawa Timur,' jelasnya.",

    "Tidak Ada Kesepakatan Politik Antara PDIP dan Anies. PDIP dan Anies belum mencapai kesepakatan politik apapun. Pertemuan ini murni merupakan bentuk silaturahim tanpa ada kesepakatan politik yang dibuat, karena belum ada keputusan apapun yang bisa diambil. 'Semuanya masih dalam proses,' ucapnya. Saat ini, PDIP sedang menyelesaikan tahap finalisasi calon yang akan maju di Pilgub Jakarta. 'Karena ini baru awal, kita tunggu saja, seperti proses di DKI dan Jawa Timur,' terangnya.",

    "Kesepakatan Politik PDIP dan Anies Masih Dalam Proses. PDIP dan Anies belum mencapai kesepakatan politik apapun hingga saat ini. Pertemuan yang dilakukan hanya sebatas silaturahim tanpa adanya kesepakatan politik, karena belum ada dasar untuk membuat kesepakatan. 'Semuanya masih dalam proses,' katanya. PDIP sedang dalam tahap finalisasi untuk menentukan calon yang akan diusung di Pilgub Jakarta. 'Ini masih awal, mari kita tunggu seperti proses di DKI dan Jawa Timur,' ungkapnya.",

    "Proses Kesepakatan Politik PDIP dan Anies Masih Berjalan. Belum ada kesepakatan politik yang dibuat antara PDIP dan Anies. Pertemuan mereka hanyalah bentuk silaturahim tanpa adanya pembahasan politik, karena belum ada keputusan untuk membuat kesepakatan. 'Semuanya masih dalam proses,' ujarnya. PDIP saat ini sedang melakukan finalisasi calon yang akan maju di Pilgub Jakarta. 'Karena ini masih tahap awal, kita tunggu saja, seperti proses di DKI dan Jawa Timur,' jelasnya.",

    "Silaturahim Tanpa Kesepakatan Politik Antara PDIP dan Anies. Pertemuan antara PDIP dan Anies belum menghasilkan kesepakatan politik. Silaturahim ini tidak membahas kesepakatan politik apapun karena belum ada posisi yang jelas untuk membuat kesepakatan. 'Semuanya masih dalam proses,' kata dia. PDIP sedang dalam proses finalisasi calon-calon yang akan diusung di Pilgub Jakarta. 'Ini masih awal, kita tunggu saja seperti proses di DKI dan Jawa Timur,' tandasnya."
]

norm_paraghraps = []
for sent in paragraphs:
    norm_sent, sentence = pp.start_words(sent)
    norm_paraghraps.append(norm_sent)

summs = summ.summarize_knn(norm_paraghraps, paragraphs)
print(summs)

