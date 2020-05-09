import os
import pickle
import numpy as np
import tensorflow as tf
from model import BiWGAN
from utils import DataInput, calc_metric, calc_auc, create_logdir, draw_prc


# hyper parameters
base_dir = os.path.dirname(os.path.realpath(__file__))
mode = 'kdd'
test_batch_size = 1024
input_dim = 121
method = 'cross-e'  # or 'fm'
weight = 0.9
degree = 'euclidean'
logdir = create_logdir(mode, method, weight, degree)
save_path = os.path.join(base_dir, logdir)


with open('{}_dataset.pkl'.format(mode), 'rb') as f:
    train_set = pickle.load(f)
    val_set = pickle.load(f)
    test_set = pickle.load(f)

x_test, y_test = test_set

print('test set', x_test.shape)

with tf.Session() as sess:
    model = BiWGAN(input_dim, method, weight, degree)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    model.restore(sess, '{}/ckpt'.format(save_path))

    ano_scores = []
    for _, batch_test_data in DataInput(x_test, test_batch_size):
        _ano_score, _, _ = model.eval(sess, batch_test_data)
        # extend
        ano_scores += list(_ano_score)
    ano_scores = np.array(ano_scores).reshape((-1, 1))

    # Highest 80% are anomalous
    prec, rec, f1 = calc_metric(y_test, ano_scores, percentile=80)

    # Calculate auc
    auprc = calc_auc(y_test, ano_scores)
    print('Prec:{:.4f}  |  Rec:{:.4f}  |  F1:{:.4f}  |  AUPRC:{:.4f}'.format(prec, rec, f1, auprc))

    # draw prc curve
    # draw_prc(y_test, ano_scores)
