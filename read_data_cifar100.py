import numpy as np
import baseline
from sklearn.preprocessing import OneHotEncoder

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

train_d = unpickle('./cifar-100-python/train')

X = np.reshape(train_d['data'], (50000,3,32,32)).transpose(0,2,3,1)
Y = train_d['fine_labels']

X_train = X[:40000]
Y_train = Y[:40000]
X_validation  = X[40000:]
Y_validation = Y[40000:]

Y_train_onehot = []

enc = OneHotEncoder(sparse=False)
# one_hot = [0 for i in range(100)]
Y_train = enc.fit_transform(np.array(Y_train).reshape(-1,1))
Y_validation = enc.fit_transform(np.array(Y_validation).reshape(-1,1))


# print Y_train.shape
baseline.train(X_train, Y_train, X_validation, Y_validation)
# import collections
# counter=collections.Counter(Y_train)
# print(counter)
