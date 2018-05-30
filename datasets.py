import pickle
import numpy as np
import string
import re
import pandas as pd
from nltk.corpus import stopwords

class Dataset:
    def __init__(self, dir, name):
        self.name=name
        self.data = None
        self.dir = dir
        self.diversities = {}
        self.stopwords = stopwords.words('english')
        self.data_df = None
        self.load_data()
        self.df_create()

    def load_data(self):
        with open(self.dir, 'rb') as infile:
            az_dataset = pickle.load(infile)
        self.data = az_dataset
        for k, v in self.data.items():
            self.data[k] = self.tokenise(v)
        self.summary_statistics()

    def load_spam(self, dir='data/spam.txt'):
        with open(dir) as infile:
            spam_data = infile.readlines()
        for item in spam_data:
            key, val = item.split(' ', 1)
            self.data[key] = self.tokenise(val)

    def tokenise(self, data_value):
        # Remove new line splitters
        to_string = ' '.join(data_value.splitlines()).lower()

        # Remove song meta information e.g. [chorus] and [Guitar Solo]
        to_string = re.sub(r'\(.*?\)|\[.*?\]|\t', '', to_string)

        # Split on spaces and remove stopwords
        listify = [item for item in to_string.split(' ') if item not in self.stopwords]

        # Filter empty list items
        listify = list(filter(None, listify))

        # Strip punctuations
        no_punc = [''.join(char for char in word if char not in string.punctuation) for word in listify]

        # Filter blank list items
        filtered = list(filter(None, no_punc))

        return filtered

    def print_data(self):
        print(self.data)

    def diversity(self):
        for k, v in self.data.items():
            self.diversities[k] = len(set(v))/len(v)

    def summary_statistics(self):
        self.song_count = len(self.data.items())
        total_words = 0
        for val in list(self.data.values()):
            total_words+=len(val)
        self.average_length = total_words/len(self.data.items())

    def print_summary(self):
        print('Number of Songs: {}\nAverage Song Length: {} words per song'.format(self.song_count,
                                                                         np.round(self.average_length, 2)))

    def df_create(self, write=False):
        data_df = pd.DataFrame(dict([(k, [v]) for k, v in self.data.items()])).transpose()
        self.data_df = data_df
