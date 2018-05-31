from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate, ShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, TensorBoard
from sklearn.model_selection import KFold


# Fit a logistic regression model to tf-idf vectors
def logistic_regression(data_obj, k):
    clf = LogisticRegression(solver='liblinear', random_state=123)
    cv = ShuffleSplit(n_splits=k, test_size=0.3, random_state=123)
    scores = cross_validate(clf, data_obj.all_vecs, data_obj.b_labels, cv=cv, scoring=['accuracy', 'precision', 'recall'],
                            return_train_score=False)
    accuracy = [np.mean(scores['test_accuracy']), 1.96*np.std(scores['test_accuracy'])/np.sqrt(k)]
    precision = [np.mean(scores['test_precision']), 1.96*np.std(scores['test_precision'])/np.sqrt(k)]
    recall = [np.mean(scores['test_recall']), 1.96*np.std(scores['test_recall'])/np.sqrt(k)]
    #roc = [np.mean(scores['test_roc_auc']), 1.96*np.std(scores['test_roc_auc'])/np.sqrt(k)]
    return accuracy, precision, recall


# Fit a random forest model to tf-idf vectors
def random_forest(data_obj, tree_count, k):
    clf = RandomForestClassifier(n_jobs=-1, n_estimators=tree_count, random_state=123)
    cv = ShuffleSplit(n_splits=k, test_size=0.3, random_state=123)
    scores = cross_validate(clf, data_obj.all_vecs, data_obj.b_labels, cv=cv, scoring=['accuracy', 'precision', 'recall'],
                            return_train_score=False)
    accuracy = [np.mean(scores['test_accuracy']), 1.96*np.std(scores['test_accuracy'])/np.sqrt(k)]
    precision = [np.mean(scores['test_precision']), 1.96*np.std(scores['test_precision'])/np.sqrt(k)]
    recall = [np.mean(scores['test_recall']), 1.96*np.std(scores['test_recall'])/np.sqrt(k)]
    return accuracy, precision, recall


def naive_bayes(data_obj, k):
    clf = MultinomialNB()
    cv = ShuffleSplit(n_splits=k, test_size=0.3, random_state=123)
    scores = cross_validate(clf, data_obj.all_vecs, data_obj.b_labels, cv=cv, scoring=['accuracy', 'precision', 'recall'],
                            return_train_score=False)
    accuracy = [np.mean(scores['test_accuracy']), 1.96*np.std(scores['test_accuracy'])/np.sqrt(k)]
    precision = [np.mean(scores['test_precision']), 1.96*np.std(scores['test_precision'])/np.sqrt(k)]
    recall = [np.mean(scores['test_recall']), 1.96*np.std(scores['test_recall'])/np.sqrt(k)]
    return accuracy, precision, recall


def svm(data_obj, k):
    clf = SVC(kernel='rbf', random_state=123)
    cv = ShuffleSplit(n_splits=k, test_size=0.3, random_state=123)
    scores = cross_validate(clf, data_obj.all_vecs, data_obj.b_labels, cv=cv, scoring=['accuracy', 'recall','precision'],
                            return_train_score=False)
    precision = [np.mean(scores['test_precision']), 1.96*np.std(scores['test_precision'])/np.sqrt(k)]
    accuracy = [np.mean(scores['test_accuracy']), 1.96*np.std(scores['test_accuracy'])/np.sqrt(k)]
    precision = [np.mean(scores['test_precision']), 1.96*np.std(scores['test_precision'])/np.sqrt(k)]
    recall = [np.mean(scores['test_recall']), 1.96*np.std(scores['test_recall'])/np.sqrt(k)]
    return accuracy, precision, recall


# Test the number of trees required for an optimal random forest
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
        self.y_enc = self._encode_labels(self.y)
        self.preprocess()
        self.y_enc_tr = self._encode_labels(self.y_tr)
        self.y_enc_te = self._encode_labels(self.y_te)
        self.model = None

    # Encode and reshape the labels object into the correct dimensions and datatype for an LSTM
    def _encode_labels(self, label):
        enc = LabelEncoder()
        y_enc = enc.fit_transform(label)
        return y_enc.reshape(-1, 1)

    # Store the words in a tokened matrix of consistent dimensions - essential for an LSTM
    def tokenise(self, input_text, lower=True):
        tokeniser = Tokenizer(self.max_words, lower=lower)
        tokeniser.fit_on_texts(input_text)
        seqs = tokeniser.texts_to_sequences(input_text)
        seqs_matrix = pad_sequences(seqs, maxlen=self.word_count)
        return seqs_matrix

    # Encode the labels as binary strings
    def preprocess(self):
        self.X_tr, self.X_te, self.y_tr, self.y_te = train_test_split(self.X, self.y_enc)

    # Define the LSTM model's structure
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
        return self.model

    # Fit a single LSTM
    def fit(self):
        self._define_model()
        X_in_tr = self.tokenise(self.X_tr)
        X_in_te = self.tokenise(self.X_te)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3)
        tboard = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
        self.model.fit(X_in_tr, self.y_enc_tr, batch_size=16, epochs=50,
                       validation_split=0.3, callbacks=[early_stopping])
        preds = self.model.predict(X_in_te)
        accuracy, precision, recall = self.evaluate_prediction(preds)
        return accuracy, precision, recall

    # Fit an LSTM with 10-fold cross-validation
    def cv(self, k=5):
        self._define_model()
        X_in = self.tokenise(self.X)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3)
        tboard = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
        clf = KerasClassifier(build_fn=lstm_model, epochs = 50, batch_size = 32, validation_split=0.3)
        folds = KFold(n_splits=k, random_state=123)
        accuracies = cross_val_score(clf, X_in, self.y_enc, cv=folds, fit_params={'callbacks': [early_stopping]})
        return [np.mean(accuracies), 1.96*(np.std(accuracies)/np.sqrt(k))]

    def test(self):
        X_in = self.tokenise(self.X_te)
        history = self.model.evaluate(X_in, self.y_te)
        print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(history[0], history[1]))

    def evaluate_prediction(self, preds):
        rounded = np.round(preds.ravel(), 0)
        tn, fp, fn, tp = confusion_matrix(y_true=self.y_enc_te, y_pred=rounded).ravel()
        try:
            accuracy = (tp+tn)/(tp+fp+tn+fn)
        except:
            accuracy = 0
        try:
            precision = tp/(tp+fp)
        except:
            precision = 0
        try:
            recall = tp/(tp+fn)
        except:
            recall = 0
        return [accuracy, 0], [precision, 0], [recall, 0]

def lstm_model():
    inputs = Input(name='inputs', shape=[750])
    layer = Embedding(200, 50, input_length=750)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256, name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1, name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs, outputs=layer)
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    return model