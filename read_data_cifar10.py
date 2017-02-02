import numpy as np
import cPickle
import word_vec
from sklearn.preprocessing import OneHotEncoder

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

embeddings = word_vec.get_vectors()

train_d = unpickle('cifar-10-batches-py/data_batch_1')
X = train_d['data']
Y = train_d['labels']
train_d = unpickle('cifar-10-batches-py/data_batch_2')
X = np.vstack((X,train_d['data']))
Y = Y + train_d['labels']
train_d = unpickle('cifar-10-batches-py/data_batch_3')
X = np.vstack((X,train_d['data']))
Y = Y + train_d['labels']
train_d = unpickle('cifar-10-batches-py/data_batch_4')
X = np.vstack((X,train_d['data']))
Y = Y + train_d['labels']
train_d = unpickle('cifar-10-batches-py/data_batch_5')
X = np.vstack((X,train_d['data']))
Y = Y + train_d['labels']

X = np.reshape(X, (50000,3,32,32)).transpose(0,2,3,1)

X_train = X[:40000]
Y_train = Y[:40000]
X_validation  = X[40000:]
Y_validation = Y[40000:]


def classification_baseline():
    import baseline
    enc = OneHotEncoder(sparse=False)
    Y_train = enc.fit_transform(np.array(Y_train).reshape(-1,1))
    Y_validation = enc.fit_transform(np.array(Y_validation).reshape(-1,1))
    baseline.train(X_train, Y_train, X_validation, Y_validation)

def classify_embedding():
    import classification_based
    class_labels = unpickle('cifar-10-batches-py/batches.meta')['label_names']
    # print class_labels
    Y_8_train = np.array(Y_train)
    X_8_train = np.array(X_train)

    removed_indices = np.where(Y_8_train!=8)
    Y_8_train = Y_8_train[removed_indices]
    X_8_train = X_8_train[removed_indices]
    removed_indices = np.where(Y_8_train!=9)
    Y_8_train = Y_8_train[removed_indices]
    X_8_train = X_8_train[removed_indices]

    enc = OneHotEncoder(sparse=False)
    Y_8_train = enc.fit_transform(np.array(Y_8_train).reshape(-1,1))

    Y_8_validation = np.array(Y_validation)
    X_8_validation = np.array(X_validation)

    removed_indices = np.where(Y_8_validation!=8)
    Y_8_validation = Y_8_validation[removed_indices]
    X_8_validation = X_8_validation[removed_indices]
    removed_indices = np.where(Y_8_validation!=9)
    Y_8_validation = Y_8_validation[removed_indices]
    X_8_validation = X_8_validation[removed_indices]

    Y_8_validation = enc.fit_transform(np.array(Y_8_validation).reshape(-1,1))

    # print X_8_validation.shape
    # print Y_8_validation.shape
    # print X_8_train.shape
    # print Y_8_train.shape

    # classification_based.train(X_8_train, Y_8_train, X_8_validation, Y_8_validation)

classify_embedding()
