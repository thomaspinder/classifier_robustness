import datasets as dt
import classifiers as clf
import preprocess as pr
import numpy as np
import pandas as pd

if __name__=='__main__':
    np.random.seed(123)

    # Load in data objects
    blur = dt.Dataset('data/Blur_lyrics.pickle', 'Blur')
    #blur.zipf_plot()
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
        result = [i]
        print('Noise Amount: {}'.format(i))
        # Preprocess
        analysis = pr.Analyser(blur, oasis, i)
        # analysis.get_summaries()
        analysis.train_test()
        analysis.get_tfidf()

        # Fit logistic regression
        print('Logistic Regression:')
        logit_acc, logit_roc = clf.svm(analysis, 10)
        print('Accuracy: {}% +/-{}\nROC: {}% +/- {}\n'.format(np.round(logit_acc[0] * 100, 2),
                                                                                      np.round(100 * logit_acc[1], 2),
                                                                                      np.round(100 * logit_roc[0], 2),
                                                                                      np.round(100 * logit_roc[1], 2)))
        result.extend(logit_acc+logit_roc)

        # Fit Random Forest
        print('Random Forest:')
        rf_acc, rf_roc = clf.random_forest(analysis, 100, 10)
        print('Accuracy: {}% +/-{}\nROC: {}% +/- {}\n'.format(np.round(rf_acc[0] * 100, 2),
                                                                                      np.round(100 * rf_acc[1], 2),
                                                                                      np.round(100 * rf_roc[0], 2),
                                                                                      np.round(100 * rf_roc[1], 2)))
        result.extend(rf_acc+rf_roc)

        # Fit Naive-Bayes

        # Fit LSTM
        print('LSTM:')
        lstm_clf = clf.LSTM_model(analysis, 750, 200)
        # lstm_clf.fit()
        lstm_acc = lstm_clf.cv()
        print('Accuracy: {} +/- {}%\n'.format(lstm_acc[0], lstm_acc[1]))
        results.extend(lstm_acc)

        # Store Results
        results.append(result)
        print('{}\n{}'.format('-'*80, '-'*80))

    # Format and write results to file
    results_df = pd.DataFrame(results, columns=['noise', 'lo_acc', 'lo_acc_se', 'lo_roc', 'lo_roc_se', 'rf_acc',
                                                'rf_acc_se', 'rf_roc', 'rf_roc_se', 'lstm_acc', 'lstm_acc_se'])
    results_df.to_csv('results/noise_results.csv', index=False)


