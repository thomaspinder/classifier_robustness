import modeller as md
import datasets as dt


if __name__=='__main__':
    eminem = dt.Dataset('data/Eminem_lyrics.pickle', 'Eminem')
    oned = dt.Dataset('data/OneDirection_lyrics.pickle', 'OneDirection')

    analysis = md.Analyser(eminem, oned)
    analysis.train_test()
    analysis.get_tfidf()