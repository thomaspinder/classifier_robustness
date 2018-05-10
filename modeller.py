from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
import logging
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

    def _create(self):
        self.data1.data_df['label'] = self.data1.name
        self.data2.data_df['label'] = self.data2.name
        self.all_data = self.data1.data_df.append(self.data2.data_df).reset_index()
        self.all_data.columns = ['track', 'lyrics', 'label']
        self.all_data = self.all_data.drop_duplicates(subset='track')
        logging.info('Full Dataset Shape: {}'.format(self.all_data.shape))

    def train_test(self, split=(0.7, 0.3)):
        X = self.all_data.drop('label', axis=1)
        y = self.all_data.label
        self.X_tr, self.X_te, self.y_tr, self.y_te = train_test_split(X, y, train_size=split[0],test_size=split[1],
                                                                      random_state=123)
        logging.info('Train Size: {}'.format(self.X_tr.shape))
        logging.info('Test Size: {}'.format(self.y_tr.shape))

    @staticmethod
    def text_extract(text_series):
        listed = [' '.join(x) for x in text_series.tolist()]
        return listed

    def split_check(self):
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
        self.split_check()
        word_vecs = TfidfVectorizer(sublinear_tf=True, strip_accents='unicode', analyzer='word', ngram_range=(1,1),
                                    stop_words='english')
        text = self.text_extract(self.all_data.lyrics)
        word_vecs.fit(text)
        X_tr_text = self.text_extract(self.X_tr.lyrics)
        X_te_text = self.text_extract(self.X_te.lyrics)
        train_vecs = word_vecs.transform(X_tr_text)
        test_vecs = word_vecs.transform(X_te_text)
        return train_vecs, test_vecs

    def get_tfidf(self):
        self.train_vecs, self.y_vecs = self.tfidf_vec()
