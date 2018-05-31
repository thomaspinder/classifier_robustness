import datasets as dt
import classifiers as clf
import preprocess as pr
import numpy as np
import pandas as pd

if __name__=='__main__':
    np.random.seed(123)

    # Load in data objects
    blur = dt.Dataset('data/Blur_lyrics.pickle', 'Blur')
    oasis = dt.Dataset('data/Oasis_lyrics.pickle', 'Oasis', True)
    analysis = pr.Analyser(blur, oasis, 0)
    # analysis.get_summaries()
    analysis.train_test()
    analysis.get_tfidf()

    # # Determine optimal number of trees
    # print('Random Forest:')
    # tree_count = clf.test_random_forest(analysis, 50, 10, True)
    # print('Optimal Tree Count: {}.'.format(tree_count))

    results = []
    for i in np.round(np.linspace(0, 0.9, 10), 1).tolist():
        print('Noise Amount: {}'.format(i))
        # Preprocess
        analysis = pr.Analyser(blur, oasis, i)
        # analysis.get_summaries()
        analysis.train_test()
        analysis.get_tfidf()

        # Fit logistic regression
        print('Logistic Regression:')
        logit_acc, logit_prec, logit_rec = clf.logistic_regression(analysis, 10)
        print('Accuracy: {}% +/-{}\nPrecsion: {}% +/- {}\nRecall: {}% +/- {}'.format(np.round(logit_acc[0] * 100, 2),
                                                                                      np.round(100 * logit_acc[1], 2),
                                                                                      np.round(100 * logit_prec[0], 2),
                                                                                      np.round(100 * logit_prec[1], 2),
                                                                                      np.round(100 * logit_rec[0], 2),
                                                                                      np.round(100 * logit_rec[1], 2)
                                                                                      ))
        result = ['logistic', i]
        result.extend(logit_acc+logit_prec+logit_rec)
        results.append(result)
        print('{}\n'.format('-'*80))

        # Fit Random Forest
        print('Random Forest:')
        rf_acc, rf_prec, rf_rec = clf.random_forest(analysis, 100, 10)
        print('Accuracy: {}% +/-{}\nPrecsion: {}% +/- {}\nRecall: {}% +/- {}\n'.format(np.round(rf_acc[0] * 100, 2),
                                                                                      np.round(100 * rf_acc[1], 2),
                                                                                      np.round(100 * rf_prec[0], 2),
                                                                                      np.round(100 * rf_prec[1], 2),
                                                                                      np.round(100 * rf_rec[0], 2),
                                                                                      np.round(100 * rf_rec[1], 2)
                                                                                      ))
        result = ['random_forest', i]
        result.extend(rf_acc+rf_prec+rf_rec)
        results.append(result)
        print('{}\n'.format('-'*80))

        # Fit Naive-Bayes
        print('Naive-Bayes:')
        nb_acc, nb_prec, nb_rec = clf.naive_bayes(analysis, 10)
        print('Accuracy: {}% +/-{}\nPrecsion: {}% +/- {}\nRecall: {}% +/- {}\n'.format(np.round(nb_acc[0] * 100, 2),
                                                                                      np.round(100 * nb_acc[1], 2),
                                                                                      np.round(100 * nb_prec[0], 2),
                                                                                      np.round(100 * nb_prec[1], 2),
                                                                                      np.round(100 * nb_rec[0], 2),
                                                                                      np.round(100 * nb_rec[1], 2)
                                                                                      ))
        result = ['naive_bayes', i]
        result.extend(nb_acc+nb_prec+nb_rec)
        results.append(result)
        print('{}\n'.format('-'*80))

        # Fit LSTM
        print('LSTM:')
        lstm_clf = clf.LSTM_model(analysis, 750, 200)
        lstm_prec, lstm_rec = lstm_clf.fit()
        lstm_acc = lstm_clf.cv()
        print('Accuracy: {} +/- {}%\nPrecision: {}\nRecall: {}'.format(lstm_acc[0]*100,
                                                                       lstm_acc[1]*100,
                                                                       np.round(lstm_prec*100, 2),
                                                                       np.round(lstm_rec*100, 2)))
        result = ['lstm', i]
        result.extend(lstm_acc)
        result.append(lstm_prec, 0, lstm_rec, 0)
        results.append(result)
        print('{}\n{}'.format('-'*80, '-'*80))

    # Format and write results to file
    results_df = pd.DataFrame(results, columns=['classifier', 'noise', 'accuracy', 'accuracy_se', 'precision',
                                                'precision_se', 'recall', 'recall_se'])
    results_df.to_csv('results/noise_results.csv', index=False)


