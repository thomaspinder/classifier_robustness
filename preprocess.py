from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from scipy.sparse import vstack
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
        self.features = None
        self.labels = None
        self.seqs_matrix = None

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
        word_vecs.fit(text)
        X_tr_text = self.text_extract(self.X_tr.lyrics)
        X_te_text = self.text_extract(self.X_te.lyrics)
        train_vecs = word_vecs.transform(X_tr_text)
        test_vecs = word_vecs.transform(X_te_text)
        all_vecs = vstack([train_vecs, test_vecs])
        return train_vecs, test_vecs, all_vecs

    def get_tfidf(self):
        self.train_vecs, self.y_vecs, self.all_vecs = self.tfidf_vec()

