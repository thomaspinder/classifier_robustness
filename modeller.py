import pickle

class Dataset:
    def __init__(self):
        self.data = None

    def load_data(self, pickle_dir):
        with open('pickle_dir', 'rb') as infile:
            az_dataset = pickle.load(infile)
        self.data = az_dataset

    def print_data(self):
        print(self.data)