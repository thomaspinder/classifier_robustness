from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from utils.key_distances import euclideanKeyboardDistance
import numpy as np
import sys
import string
import logging
import multiprocessing
import gensim.models.word2vec as w2v
from copy import deepcopy
import matplotlib.pyplot as plt
from collections import Counter
from gensim.models import KeyedVectors
logging.basicConfig(filename='modelling.log', level=logging.INFO,
                    format= '[%(asctime)s] %(levelname)s - %(message)s',
                    filemode='w')


class Analyser:
    def __init__(self, dataset1, dataset2, noise=0):
        self.data1 = dataset1
        self.data2 = dataset2
        self.labels = (self.data1.name, self.data2.name)
        self.all_data = None
        self._create()
        self.X_tr = None
        self.X_te = None
        self.y_tr = None
        self.y_te = None
        self.features = None
        self.labels = None
        self.b_labels = None
        self.seqs_matrix = None
        self.letters = list(string.ascii_letters)
        if 0 <= noise < 1:
            self.add_noise(noise)
        else:
            raise ValueError('Noise amount must be between 0 and 1 with 0 indicating no noise.')
        self.noise_amount = noise

    def _create(self):
        """
        Join up the two datasets and label the observations.
        :return: DataFrame
        """
        self.data1.data_df['label'] = self.data1.name
        self.data2.data_df['label'] = self.data2.name
        self.all_data = self.data1.data_df.append(self.data2.data_df).reset_index()
        self.all_data.columns = ['track', 'lyrics', 'label']
        self.all_data = self.all_data.drop_duplicates(subset='track')
        self.all_data = self.all_data[self.all_data.lyrics.apply(len) > 10]
        self.backup = deepcopy(self.all_data)
        logging.info('Full Dataset Shape: {}'.format(self.all_data.shape))

    def train_test(self, split=(0.7, 0.3)):
        """
        Partition data out into features and labels, then split data into training/testing set.
        :param split: Train/Test split proportion
        :return: 4 DataFrames
        """
        X = self.all_data.drop('label', axis=1)
        y = self.all_data.label
        self.X_tr, self.X_te, self.y_tr, self.y_te = train_test_split(X, y, train_size=split[0],test_size=split[1],
                                                                      random_state=123)
        self.features = X
        self.labels = y
        labels_uni = np.unique(y)
        self.b_labels = np.array([0 if item == labels_uni[0] else 1 for item in y])
        logging.info('Train Size: {}'.format(self.X_tr.shape))
        logging.info('Test Size: {}'.format(self.y_tr.shape))

    @staticmethod
    def text_extract(text_series):
        """
        Split a scraped set of lyrics out into a list whereby each element is a single word
        :param text_series: Series of lyric sets
        :return: List of words
        """
        listed = [' '.join(x) for x in text_series.tolist()]
        return listed

    def split_check(self):
        """
        Check data has been correctly partitioned
        :return: Bool
        """
        if self.X_tr is None or self.X_te is None or self.y_te is None or self.y_tr is None:
            print('Data not split into test/train set yet.')
            split_bool = input('Should this data be split (y/n)?  ')
            if split_bool == 'y':
                self.train_test()
                print('Data split in test/train set.')
            else:
                print('Ending, please split data before fitting vectors.')
                sys.exit()

    def tfidf_vec(self):
        """
        Create tf-idf vectors from tokenised words
        :return: Matrix of vectors
        """
        self.split_check()
        word_vecs = TfidfVectorizer(sublinear_tf=True, strip_accents='unicode', analyzer='word', ngram_range=(1,1),
                                    stop_words='english', max_features=50, smooth_idf=True)
        text = self.text_extract(self.all_data.lyrics)
        all_vecs = word_vecs.fit_transform(text)
        return all_vecs

    def get_tfidf(self):
        """
        Calculate tf-idf vectors
        """
        self.all_vecs = self.tfidf_vec()

    def add_noise(self, amount = 0.9):
        """
        Synthetically induce noise to the data
        """
        clean = self.all_data.lyrics.tolist()
        noisy = [self._generate_noise(lyric, amount) for lyric in clean]
        self.all_data.lyrics = noisy

    def _generate_noise(self, string, amount):
        """
        Noise Generating function for a given string and the desired noise amount
        """
        noise_amount = np.ceil(amount*len(string)).astype(int)
        candidates = np.random.choice(len(string), noise_amount)
        for i in candidates:
            if len(string[i]) < 2:
                pass
            else:
                string[i] = np.random.choice([self._remove, self._replace, self._add_character])(string[i])
        return string

    def _replace(self, string):
        """
        Replace a random character within a string a replace with a close by keyboard key
        """
        index = np.random.randint(0, len(string))
        old = string[index]
        character = np.random.choice(self.letters)
        distance = euclideanKeyboardDistance(old, character)
        while distance < 1.2:
            character = np.random.choice(self.letters)
            distance = euclideanKeyboardDistance(old, character)
        string = string[:index] + character + string[index+1:]
        return string

    def _add_character(self, string):
        """
        Add an aditional character to a random point within the string
        """
        index = np.random.randint(0, len(string))
        character = np.random.choice(self.letters)
        return string[:index] + character + string[index:]

    @staticmethod
    def _remove(string):
        """
        Delete a random character from the string
        """
        index = np.random.randint(0, len(string))
        return string[:index] + string[index+1:]

    def reset_data(self):
        self.all_data = self.backup

    def get_summaries(self):
        """
        Print summary statistics
        """
        lengths = self.all_data.lyrics.apply(len)
        print('Track Statistics:\nMax Length: {}\nShortest Length: {}\nMean Length: {}'.format(max(lengths),
                                                                                                min(lengths),
                                                                                                np.mean(lengths)))
        print('')
        print('Longest Track: {}'.format(self.all_data.track[np.argmax(lengths)].replace(' lyrics', '')))
        print('Shortest Track: {}'.format(self.all_data.track[np.argmin(lengths)].replace(' lyrics', '')))
        print('-'*80)
        artists = self.all_data.label.value_counts()
        print(artists)