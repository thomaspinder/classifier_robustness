import requests
from bs4 import BeautifulSoup, SoupStrainer
from lxml import html
import time
import numpy as np
import pickle


class Lyrics:
    def __init__(self, artist_name, verbose=False, max_requests=20):
        self.meta_name = artist_name
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
        self.session = None
        self.max_requests = max_requests
        self.setup_session()

    def setup_session(self):
        s = requests.session()
        a = requests.adapters.HTTPAdapter(max_retries=self.max_requests)
        s.mount('https://', a)
        s.mount('http://', a)
        self.session = s
        print(s)
        if self.verbosity:
            print('Session Successfully Initialised')

    def scrape(self):
        """
        Carry out the lifting of the scraper, collecting the track links and scraping the respective lyrics. The results
        are then stored in a dictionay and pickled for future used.
        :return: Pickled Dictionary
        """
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
        """
        From a given url, a list of the artist's song titles urls are gathered
        :param url: Artists url
        :return: List of urls containing lyrics
        """
        title_html = self.session.get(url, headers=self.headers).content
        soup = BeautifulSoup(title_html, 'html.parser', parse_only=SoupStrainer('div', {'id': 'listAlbum'}))
        self.links = [a for a in soup.find_all('a', href=True)]

    def print_results(self):
        """
        Print the amount of song titles that have been found
        """
        print('{} Tracks Found'.format(len(self.links)))

    def _get_tracks(self):
        """
        From the stored list of lyric urls, the track's respective lyrics are now scraped. This is made slightly harder
        due to the lack of css on azlyric's page, fortunately the lyric data can be identified through xpath notation
        though. AzLyrics block an IP for making too many requests in a short space of time, so a pause of a randomly
        generated length is made in between requests to prevent IP detection.
        :return:
        """
        for link in self.links:
            try:
                content = self.session.get(self.base_url+link['href'].replace('..', ''), headers=self.headers).content
                tree = html.fromstring(content)
                title = tree.xpath('/html/body/div[3]/div/div[2]/b/text()')[0]
                lyrics = ' '.join(tree.xpath('/html/body/div[3]/div/div[2]/div[5]/text()'))
                self.lyrics[title] = lyrics
                random_pause = round(np.random.uniform(0, 10, 1), 0)
                if self.verbosity and (len(self.lyrics.items())*100) % len(self.links) == 0:
                    print('{}% Complete'.format(round(len(self.lyrics.items())*100/len(self.links))))
                time.sleep(random_pause)
            except:
                print('No server response.')
                break

    def pickle_results(self):
        """
        Pickle and store the results for use in analysis later in the pipeline.
        :return:
        """
        with open('data/{}_lyrics.pickle'.format(self.meta_name), 'wb') as outfile:
            pickle.dump(self.lyrics, outfile, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    scraper = Lyrics('Queen', verbose=True)
    scraper.scrape()
    scraper.print_results()
