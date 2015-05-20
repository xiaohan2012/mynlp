import pandas as pd
import cPickle as pkl
from scipy.io import loadmat
from scipy.spatial.distance import cdist

Y_path = '/cs/puls/Experiments/hxiao-test/label-data.mat'
label_path = '/cs/puls/Experiments/hxiao-test/sector-labels'

y = loadmat(Y_path)['labelData']
with open(label_path, 'r') as f:
    y_names = map(lambda l: l.strip(), f.readlines())

y = y.T.todense()

distance_metrices = ['cityblock', 'seuclidean', 'cosine',
                     'correlation', 'hamming', 'jaccard',
                     'canberra', 'braycurtis', 'chebyshev']

columns = ['Electronics', 'Construction']

for metric in distance_metrices:
    print "###" * 10
    print metric
    print "###" * 10
    print ""

    distmat = cdist(y, y, metric)
    pkl.dump(distmat, open('data/pickle/%s.pkl' % metric, 'w'))

    df = pd.DataFrame(data=distmat,
                      columns=y_names,
                      index=y_names)
    with open('result/%s.txt', 'w') as f:
        for col in columns:
            ans = df[[col]].sort(columns=col, axis=0, ascending=1)
            f.write(str(ans))
            print ans
