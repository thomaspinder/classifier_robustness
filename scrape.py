import requests
from bs4 import BeautifulSoup, SoupStrainer
from lxml import html
import time
import numpy as np
import pickle


class Lyrics:
    def __init__(self, artist_name, verbose = False):
        self.url = 'https://www.azlyrics.com/q/{}.html'.format(artist_name.lower())
        self.links = None
        self.lyrics = {}
        self.headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.10; rv:30.0) " +
                                      "Gecko/20100101 Firefox/30.0",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.5",
                        "Accept-Encoding": "gzip, deflate",
                        "Connection": "keep-alive"}
        self.verbosity = verbose
        self.base_url = 'https://www.azlyrics.com'


    def scrape(self):
        if self.verbosity:
            print('Scraping {}'.format(self.url))
        self._get_titles(self.url)
        if self.verbosity:
            self.print_results()
            print('Collecting Individual Tracks.')
        self._get_tracks()
        if self.verbosity:
            print('Pickling Results')
        self.pickle_results()

    def _get_titles(self, url):
        html = requests.get(url, headers=self.headers).content
        soup = BeautifulSoup(html, 'html.parser', parse_only=SoupStrainer('div', {'id': 'listAlbum'}))
        self.links = [a for a in soup.find_all('a', href=True)]
        self.links = [self.links[1], self.links[2], self.links[3]]

    def print_results(self):
        print('{} Tracks Found'.format(len(self.links)))

    def _get_tracks(self):
        for link in self.links:
            content = requests.get(self.base_url+link['href'].replace('..', ''), headers=self.headers).content
            tree = html.fromstring(content)
            title = tree.xpath('/html/body/div[3]/div/div[2]/b/text()')[0]
            lyrics = ' '.join(tree.xpath('/html/body/div[3]/div/div[2]/div[5]/text()'))
            self.lyrics[title] = lyrics
            random_pause = np.round(np.random.uniform(0, 10, 1), 0)
            if self.verbosity:
                print('Pausing for {}secs.'.format(random_pause))
            time.sleep(random_pause)

    def pickle_results(self):
        with open('{}_lyrics.txt', 'wb') as outfile:
            pickle.dump(self.lyrics, outfile)

if __name__ == '__main__':
    scraper = Lyrics('Queen', verbose=True)
    scraper.scrape()
    scraper.print_results()
