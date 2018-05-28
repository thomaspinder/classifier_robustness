from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import scikitplot as skplt
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping


def random_forest(data_obj, max_trees = 400, increment = 10, plot=True, k=10):
    means = []
    sds = []
    for i in range(10, max_trees, increment):
        if i%100==0:
            print('Fitting {}th tree.'.format(i))
        rf_clf = RandomForestClassifier(n_jobs=-1, n_estimators=1)
        scores = cross_val_score(rf_clf, data_obj.all_vecs, data_obj.labels, cv=k)
        means.append(np.mean(scores))
        sds.append(np.std(scores) / np.sqrt(len(data_obj.labels)))
    if plot:
        plt.errorbar(list(range(10, max_trees, increment)), means, sds)
        plt.title('Random Forest Classifier Performance')
        plt.ylabel('Accuracy')
        plt.xlabel('Number of Trees')
        plt.savefig('plots/{}_{}_rf.png'.format(data_obj.data1.name, data_obj.data2.name))

class LSTM_model:
    def __init__(self, data_obj, word_count, max_word):
        self.X = [' '.join(lyric) for lyric in data_obj.features.lyrics.tolist()]
        self.y = data_obj.labels
        self.X_tr = None
        self.X_te = None
        self.y_tr = None
        self.y_te = None
        self.word_count = word_count
        self.max_words = max_word
        self.y_enc = self._encode_labels()
        self.model = None
        self.preprocess()

    def _encode_labels(self):
        enc = LabelEncoder()
        y_enc = enc.fit_transform(self.y)
        return y_enc.reshape(-1, 1)

    def tokenise(self, input_text, lower=True, matrix_dim = 200):
        """
        :param top_n: The top n most popular words to keen.
        :return:
        """
        tokeniser = Tokenizer(self.max_words, lower=lower)
        tokeniser.fit_on_texts(input_text)
        seqs = tokeniser.texts_to_sequences(input_text)
        seqs_matrix = pad_sequences(seqs, maxlen=self.word_count)
        return seqs_matrix

    def preprocess(self):
        self._encode_labels()
        self.X_tr, self.X_te, self.y_tr, self.y_te = train_test_split(self.X, self.y_enc)

    def _define_model(self):
        inputs = Input(name='inputs', shape=[self.word_count])
        layer = Embedding(self.max_words, 50, input_length=self.word_count)(inputs)
        layer = LSTM(64)(layer)
        layer = Dense(256, name='FC1')(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(0.5)(layer)
        layer = Dense(1, name='out_layer')(layer)
        layer = Activation('sigmoid')(layer)
        self.model = Model(inputs=inputs, outputs=layer)

    def fit(self):
        self._define_model()
        self.model.compile(loss='binary_crossentropy', optimizer=RMSprop(),metrics=['accuracy'])
        X_in = self.tokenise(self.X_tr)
        self.model.fit(X_in, self.y_tr, batch_size=16, epochs=10,
                  validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)])

    def test(self):
        X_in = self.tokenise(self.X_te)
        history = self.model.evaluate(X_in, self.y_te)
        print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(history[0], history[1]))