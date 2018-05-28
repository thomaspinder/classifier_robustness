from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from utils.key_distances import euclideanKeyboardDistance
import numpy as np
import sys
import random
import string
import logging
import multiprocessing
import gensim.models.word2vec as w2v
from copy import deepcopy
from gensim.models import KeyedVectors
logging.basicConfig(filename='modelling.log', level=logging.INFO,
                    format= '[%(asctime)s] %(levelname)s - %(message)s',
                    filemode='w')


class Analyser:
    def __init__(self, dataset1, dataset2):
        self.data1 = dataset1
        self.data2 = dataset2
        self.all_data = None
        self._create()
        self.X_tr = None
        self.X_te = None
        self.y_tr = None
        self.y_te = None
        self.features = None
        self.labels = None
        self.seqs_matrix = None
        self.letters = list(string.ascii_letters)

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
                                    stop_words='english')
        text = self.text_extract(self.all_data.lyrics)
        all_vecs = word_vecs.fit_transform(text)
        # X_tr_text = self.text_extract(self.X_tr.lyrics)
        # X_te_text = self.text_extract(self.X_te.lyrics)
        # train_vecs = word_vecs.transform(X_tr_text)
        # test_vecs = word_vecs.transform(X_te_text)
        # old = vstack([train_vecs, test_vecs])
        return all_vecs # train_vecs, test_vecs, all_vecs

    def get_tfidf(self):
        self.all_vecs = self.tfidf_vec()

    def add_noise(self, amount = 0.8):
        clean = self.all_data.lyrics.tolist()
        print(clean[0])
        noisy = [self._generate_noise(lyric, amount) for lyric in clean]
        print(noisy[0])

    def _generate_noise(self, string, amount):
        noise_amount = np.ceil(amount*len(string)).astype(int)
        candidates = np.random.choice(len(string), noise_amount)
        for i in candidates:
            if len(string[i]) < 2:
                pass
            else:
                string[i] = np.random.choice([self._remove, self._replace, self._add_character])(string[i])
        return string

    def _replace(self, string):
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
        index = np.random.randint(0, len(string))
        character = np.random.choice(self.letters)
        return string[:index] + character + string[index:]

    @staticmethod
    def _remove(string):
        index = np.random.randint(0, len(string))
        return string[:index] + string[index+1:]

    def reset_data(self):
        self.all_data = self.backup

    def _w2v_vec(self, text, epochs = 800, save=True):
        w2v_model = w2v.Word2Vec(
            sg = 0,
            seed = 123,
            workers=multiprocessing.cpu_count(),
            min_count=3,
            window=10,
            hs=0,
            sample=1e-3
        )
        w2v_model.build_vocab(text)
        print('Word2Vec Vocabulary Size: {}'.format(len(w2v_model.wv.vocab)))
        w2v_model.train(text, epochs=epochs, total_examples=len(text))
        w2v_vectors = w2v_model.wv
        if save:
            w2v_vectors.save('data/w2v_vectors.txt')

    def load_vectors(self, vector_dir='/data/w2v_vectors.txt'):
        w2v_vectors = KeyedVectors.load_word2vec_format(vector_dir, binary=False)
        self.w_vecs = w2v_vectors

    def get_w2v(self):
        self.w_vecs = self._w2v_vec(self.all_data.lyrics)