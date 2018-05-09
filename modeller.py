import pickle

class Dataset:
    def __init__(self, dir):
        self.data = None
        self.dir = dir

    def load_data(self):
        with open(self.dir, 'rb') as infile:
            az_dataset = pickle.load(infile)
        self.data = az_dataset
        print('Successfully Loaded {} Tracks'.format(len(self.data.items())))

    def print_data(self):
        print(self.data)

if __name__ == '__main__':
    x = Dataset('data/Queen_lyrics.pickle')
    x.load_data()
