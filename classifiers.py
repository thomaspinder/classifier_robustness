from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import scikitplot as skplt
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, TensorBoard
from sklearn.model_selection import KFold

def logistic_regression(data_obj, k):
    clf = LogisticRegression(solver='liblinear', random_state=123)
    scores = cross_validate(clf, data_obj.all_vecs, data_obj.b_labels, cv=k, scoring=['accuracy', 'roc_auc'],
                            return_train_score=False)
    accuracy = [np.mean(scores['test_accuracy']), 1.96*np.std(scores['test_accuracy'])/np.sqrt(k)]
    roc = [np.mean(scores['test_roc_auc']), 1.96*np.std(scores['test_roc_auc'])/np.sqrt(k)]
    return accuracy, roc

def random_forest(data_obj, tree_count, k):
    clf = RandomForestClassifier(n_jobs=-1, n_estimators=tree_count, random_state=123)
    scores = cross_validate(clf, data_obj.all_vecs, data_obj.b_labels, cv=k, scoring=['accuracy', 'roc_auc'],
                            return_train_score=False)
    accuracy = [np.mean(scores['test_accuracy']), 1.96*np.std(scores['test_accuracy'])/np.sqrt(k)]
    roc = [np.mean(scores['test_roc_auc']), 1.96*np.std(scores['test_roc_auc'])/np.sqrt(k)]
    return accuracy, roc

def svm(data_ob, k):
    pass


def test_random_forest(data_obj, max_trees = 400, increment = 10, plot=True, k=10):
    means = []
    sds = []
    for i in range(10, max_trees, increment):
        if i%100==0:
            print('Fitting {}th tree.'.format(i))
        rf_clf = RandomForestClassifier(n_jobs=-1, n_estimators=i, random_state=123)
        scores = cross_val_score(rf_clf, data_obj.all_vecs, data_obj.b_labels, cv=k)
        means.append(np.mean(scores))
        sds.append(np.std(scores) / np.sqrt(len(data_obj.labels)))
    if plot:
        plt.errorbar(list(range(10, max_trees, increment)), means, sds)
        plt.title('Random Forest Classifier Performance')
        plt.ylabel('Accuracy')
        plt.xlabel('Number of Trees')
        plt.savefig('plots/{}_{}_{}_noise_rf.png'.format(data_obj.data1.name, data_obj.data2.name,
                                                         data_obj.noise_amount))
    return (np.argmax(means)+1)*10


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

    def tokenise(self, input_text, lower=True):
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
        self.model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

    def fit(self):
        self._define_model()
        X_in = self.tokenise(self.X_tr)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5)
        tboard = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
        self.model.fit(X_in, self.y_tr, batch_size=16, epochs=100,
                  validation_split=0.2, callbacks=[early_stopping, tboard])

    def cv(self, k=10):
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5)
        tboard = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
        clf = KerasClassifier(build_fn=self._define_model, epochs = 50, batch_size = 12,
                              callbacks=[early_stopping, tboard])
        folds = KFold(n_splits=k, random_state=123)
        accuracies = cross_val_score(clf, self.tokenise(self.tokenise(self.X)), self.y, cv=k)
        return np.mean(accuracies), 1.96*(np.std(accuracies)/np.sqrt(k))

    def test(self):
        X_in = self.tokenise(self.X_te)
        history = self.model.evaluate(X_in, self.y_te)
        print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(history[0], history[1]))

    def get_embeddings(self):
        pass