from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import scikitplot as skplt

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

def cnn:
    pass