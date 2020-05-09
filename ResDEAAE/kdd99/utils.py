import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support, auc


class DataInput(object):
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data = data
        self.nb_batch = len(self.data) // self.batch_size
        if self.nb_batch * self.nb_batch < len(self.data):
            self.nb_batch += 1
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i == self.nb_batch:
            raise StopIteration
        batch_data = self.data[self.i*self.batch_size: min((self.i + 1)*self.batch_size,
                                                           len(self.data))]
        self.i += 1
        return self.i, batch_data


def _shuffle(x, y):
    assert len(x) == len(y)
    s = np.random.permutation(len(x))
    return x[s], y[s]


def _split_dataset(x, y, percentage=0.2):
    x, y = _shuffle(x, y)
    size = int(len(x) * percentage)
    x_test, y_test = x[:size], y[:size]
    x_train, y_train = x[size:], y[size:]
    return (x_test, y_test), (x_train, y_train)


def create_logdir(mode, method, weight, degree):
    return 'train_logs/{}/{}/{}/{}'.format(mode, method, weight, degree)


def calc_auc(y, scores):
    precision, recall, thresholds = precision_recall_curve(y, scores)
    _auc = auc(x=recall, y=precision)
    return _auc


def calc_metric(y, scores, percentile=80):
    y = np.squeeze(y).astype(np.float32)
    scores = np.squeeze(scores).astype(np.float32)
    per = np.percentile(scores, percentile)
    y_pred = (scores > per) * 1.
    prec, rec, f1, _ = precision_recall_fscore_support(y_true=y, y_pred=y_pred, average='binary')
    return prec, rec, f1


def draw_prc(y, scores):
    precision, recall, thresholds = precision_recall_curve(y, scores)
    prc_auc = auc(x=recall, y=precision)
    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve: AUC=%0.4f'% (prc_auc))
    plt.savefig('prc.jpg')
    plt.close()
