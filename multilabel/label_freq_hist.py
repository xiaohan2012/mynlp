# coding: utf-8
import cPickle as pkl
from matplotlib import pyplot as plt
from label_stat import _to_dense_array
from scipy.io import loadmat

y=pkl.load(open('data/del_y_train.pkl'))
# y = loadmat('/cs/puls/Experiments/hxiao-test/label-data.mat')['labelData']

y = _to_dense_array(y)

y = y.sum(axis=0)

fig=plt.figure()
plt.hist(y)
# fig.savefig('figures/sector_label_frequency_histogram.png')
fig.savefig('figures/delicious_label_frequency_histogram.png')

