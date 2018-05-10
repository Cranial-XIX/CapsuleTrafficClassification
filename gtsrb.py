import csv
import cv2
import numpy as np
import pickle


def load_data(root="data/GTSRB", seed=0):
    # make train set
    np.random.rand(seed)
    classes = np.arange(0, 43)
    X_tr, y_tr, X_te, y_te = [], [], [], []
    for c in range(43):
        class_name = format(classes[c], '05d')
        prefix = root + '/Train/' + class_name + '/'
        f = open(prefix + 'GT-' + class_name + '.csv')
        reader = csv.reader(f, delimiter=';')
        next(reader, None)
        X_tmp, y_tmp = [], []
        for row in reader:
            im = cv2.imread(prefix + row[0])
            im = im[np.int(row[4]):np.int(row[6]), 
                    np.int(row[3]):np.int(row[5]), :]
            X_tmp.append(im)
            y_tmp.append(c)
        l = len(y_tmp)
        X_tr += X_tmp[l//5:]
        y_tr += y_tmp[l//5:]
        X_te += X_tmp[:l//5]
        y_te += y_tmp[:l//5]
        f.close()
    size = (32, 32)
    X_tr = [cv2.resize(x, size) for x in X_tr]
    X_te = [cv2.resize(x, size) for x in X_te]
    X_tr, y_tr = np.array(X_tr).astype(np.float64), np.array(y_tr)
    X_te, y_te = np.array(X_te).astype(np.float64), np.array(y_te)
    X_tr, X_te = (X_tr - 128) / 128, (X_te - 128) / 128
    i = np.random.permutation(len(y_tr))
    X_tr, y_tr = X_tr[i], y_tr[i]
    i = np.random.permutation(len(y_te))
    X_te, y_te = X_te[i], y_te[i]
    return X_tr, y_tr, X_te, y_te

def make_data():
    X_tr, y_tr, X_te, y_te = load_data()
    pickle.dump((X_tr, y_tr), open('data/train.p', 'wb'))
    pickle.dump((X_te, y_te), open('data/test.p', 'wb'))

if __name__ == "__main__":
    make_data()