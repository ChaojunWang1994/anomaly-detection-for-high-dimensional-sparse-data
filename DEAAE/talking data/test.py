import os
import pickle
import numpy as np
import tensorflow as tf
from model import BiWGAN
from utils import DataInput, calc_auroc, calc_metric, create_logdir, _split_dataset, draw_prc

base_dir = os.path.dirname(os.path.realpath(__file__))
mode = 'talkingdata'
test_batch_size = 1024
method = 'fm'  # 'fm' or 'cross-e'
weight = 0.9
degree = 1
logdir = create_logdir(mode, method, weight, degree)
save_path = os.path.join(base_dir, logdir)
ano_size = 228423


with open('{}_dataset.pkl'.format(mode), 'rb') as f:
    train_set = pickle.load(f)
    test_set = pickle.load(f)
    count_list = pickle.load(f)

x_test, y_test = test_set
(x, y), _ = _split_dataset(x_test, y_test, percentage=0.2)
print('test set:', x.shape)


def evaluation(sess, model):
    ano_scores = []
    for _, batch_data in DataInput(x, test_batch_size):
        _ano_score = model.eval(sess, batch_data)
        # Extend
        ano_scores += list(_ano_score)
    ano_scores = np.array(ano_scores).reshape((-1, 1))

    with open('scores.pkl', 'wb') as f:
        pickle.dump((y, ano_scores), f, pickle.HIGHEST_PROTOCOL)

    # Calculate auc
    auroc = calc_auroc(y, ano_scores)
    print('Eval_auroc:{:.4f}'.format(auroc))
    prec, rec, f1 = calc_metric(y, ano_scores)
    print('Prec:{:.4f}\tRec:{:.4f}\tF1:{:.4f}\n'.format(prec, rec, f1))

    draw_prc(y, ano_scores, key='DEAAE_'+method)


with tf.Session() as sess:
    model = BiWGAN(count_list, method, weight=weight, degree=degree)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    model.restore(sess, 'train_logs_fm_0.9156/talkingdata/fm/0.9/1/ckpt')
    # model.restore(sess, 'train_logs_ce_0.9169/talkingdata/cross-e/0.9/1/ckpt')

    evaluation(sess, model)
