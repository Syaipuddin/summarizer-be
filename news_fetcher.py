from urllib.request import Request, urlopen
from bs4 import BeautifulSoup


class NewsFetcher():

    def get_news(self, urls):

        all_paraghraps = []

        for i in urls:

            req = Request(
                url=i,
                headers={'User-Agent': 'Mozilla/5.0'}
            )

            html = urlopen(req)
            contents = html.read()
            paragraph = []

            soup = BeautifulSoup(contents, 'html.parser')

            for script in soup(["script", "style", "header", "footer", "nav", "aside"]):
                script.decompose()

            for a in soup.find_all('a', href=True):
                a.extract()

            for a in soup.find_all('p', class_='baca'):
                a.extract()

            for data in soup.find_all('p'):
                paragraph.append(data.text)

            all_paraghraps.append(' '.join(paragraph))

        return all_paraghraps