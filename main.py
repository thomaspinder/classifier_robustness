import preprocess as pr
import datasets as dt
import classifiers as clf


if __name__=='__main__':
    # Load in data objects
    eminem = dt.Dataset('data/Eminem_lyrics.pickle', 'Eminem')
    queen = dt.Dataset('data/Queen_lyrics.pickle', 'Queen')

    # Preprocess
    analysis = pr.Analyser(eminem, queen)
    analysis.train_test()
    analysis.get_tfidf()

    # Preprocess LSTM
    lstm_clf = clf.LSTM_model(analysis, 750, 200)
    lstm_clf.fit()


    # Fit Random Forest
    # clf.random_forest(analysis, max_trees=5)