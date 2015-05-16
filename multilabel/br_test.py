import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

# from sklearn.datasets import make_multilabel_classification
# from sklearn.preprocessing import LabelBinarizer

X_path = '/cs/puls/Experiments/hxiao-test/feature-data.mat'
Y_path = '/cs/puls/Experiments/hxiao-test/label-data.mat'

X = loadmat(X_path)['featureData']
y = loadmat(Y_path)['labelData']

print OneVsRestClassifier(LinearSVC(random_state=0), n_jobs=-1).fit(X, y).predict(X)


