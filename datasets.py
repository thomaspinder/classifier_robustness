import pickle
import numpy as np
import string
import re
import pandas as pd
from nltk.corpus import stopwords
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt


class Dataset:
    def __init__(self, dir, name, zipf=False):
        self.name=name
        self.data = None
        self.dir = dir
        self.diversities = {}
        self.stopwords = stopwords.words('english')
        self.data_df = None
        self.load_data()
        self.df_create()
        self.plot_zipf(zipf)

    def plot_zipf(self, zipf):
        """
        Plot the word frequency distribution
        """
        words = Counter([word for lyric in self.data_df.ix[:, 0].tolist() for word in lyric])
        vals = np.array(list(words.values()))
        if zipf:
            sns.kdeplot(vals, bw=0.8, label=self.name, lw=3, linestyle="--")
            plt.ylabel('Density')
            plt.xlim(-2, 60)
            plt.xlabel('Counts')
            plt.title("Density Plots of Each Dataset's Word Count Distribution")
            plt.legend(loc='upper right')
            plt.savefig('plots/zipf_density.png')
        else:
            sns.kdeplot(vals, bw=0.8, label=self.name, lw=3)

    def load_data(self):
        """
        Load the data into a dictionary
        """
        with open(self.dir, 'rb') as infile:
            az_dataset = pickle.load(infile)
        self.data = az_dataset
        for k, v in self.data.items():
            self.data[k] = self.tokenise(v)
        self.summary_statistics()

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
        """
        Measure the datasets lexical diversity
        """
        for k, v in self.data.items():
            self.diversities[k] = len(set(v))/len(v)

    def summary_statistics(self):
        """
        Calculate summary statistics
        """
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
