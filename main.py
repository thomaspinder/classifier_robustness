import preprocess as pr
import datasets as dt
import classifiers as clf
import numpy as np
from sklearn.linear_model import LogisticRegression


if __name__=='__main__':
    # Load in data objects
    eminem = dt.Dataset('data/Eminem_lyrics.pickle', 'Eminem')
    queen = dt.Dataset('data/Queen_lyrics.pickle', 'Queen')

    # Preprocess
    analysis = pr.Analyser(eminem, queen)
    analysis.add_noise()
    # analysis.get_summaries()
    analysis.train_test()
    analysis.get_tfidf()

    # Fit logistic regression
    print('Logistic Regression:')
    logit_acc, logit_roc = clf.logistic_regression(analysis, 10)
    print('Accuracy: {}% +/-{}\nROC: {}% +/- {}\n{}'.format(np.round(logit_acc[0] * 100, 2),
                                                                                  np.round(100 * logit_acc[1], 2),
                                                                                  np.round(100 * logit_roc[0], 2),
                                                                                  np.round(100 * logit_roc[1], 2),
                                                                                  '-' * 80))

    # Fit Random Forest
    print('Random Forest:')
    tree_count = clf.test_random_forest(analysis, 200, 10, True)
    print('Fitting on {} trees.'.format(tree_count))
    rf_acc, rf_roc = clf.random_forest(analysis, tree_count, 10)
    print('Accuracy: {}% +/-{}\nROC: {}% +/- {}\n{}'.format(np.round(rf_acc[0] * 100, 2),
                                                                                  np.round(100 * rf_acc[1], 2),
                                                                                  np.round(100 * rf_roc[0], 2),
                                                                                  np.round(100 * rf_roc[1], 2),
                                                                                  '-' * 80))

    # Preprocess LSTM
    lstm_clf = clf.LSTM_model(analysis, 750, 200)
    lstm_clf.fit()
    # lstm_clf.cv()

