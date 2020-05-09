import os
import time
import pickle
import numpy as np
import tensorflow as tf
import sys
from model import BiWGAN
from build_dataset import build_dataset
from utils import DataInput, calc_auroc, calc_metric, create_logdir, _shuffle, _split_dataset

base_dir = os.path.dirname(os.path.realpath(__file__))
mode = 'talkingdata'
train_batch_size = 64
test_batch_size = 1024
nb_epochs = 300
print_iter = 500
d_g_iter = 6  # (D's training epochs / G's training epochs = d_g_iter - 1)
learning_rate = 1e-4
method = 'fm'  # or 'cross-e'
weight = 0.9
degree = 1
logdir = create_logdir(mode, method, weight, degree)
save_path = os.path.join(base_dir, logdir)
best_auroc = 0.
best_f1 = 0.
ano_size = 228423
size = 100000

if not os.path.exists(save_path):
    os.makedirs(save_path)

if not os.path.exists('{}_dataset.pkl'.format(mode)):
    build_dataset(mode)

# prepare data
with open('{}_dataset.pkl'.format(mode), 'rb') as f:
    train_set = pickle.load(f)
    test_set = pickle.load(f)
    count_list = pickle.load(f)

x_train, y_train = train_set
x_test, y_test = test_set

x, y = x_train[:-ano_size], y_train[:-ano_size]  # for training
x_train_ano, y_train_ano = x_train[-ano_size:], y_train[-ano_size:]
(x_val, y_val), _ = _split_dataset(x_test, y_test, percentage=0.1)

x, y = x[:size], y[:size]
# contaminated data
# x = np.concatenate((x, x_train_ano[:4167]), axis=0)
# y = np.concatenate((y, y_train_ano[:4167]), axis=0)
print('training set:', x.shape)
print('validation set:', x_val.shape)
print(np.sum(y))


def _eval(sess, model, test_data, label):
    ano_scores = []
    for _, batch_data in DataInput(test_data, test_batch_size):
        _ano_score = model.eval(sess, batch_data)
        # Extend
        ano_scores += list(_ano_score)

    ano_scores = np.array(ano_scores).reshape((-1, 1))

    # Calculate auroc
    auroc = calc_auroc(label, ano_scores)
    # Calculate metric
    prec, rec, f1 = calc_metric(label, ano_scores)

    global best_auroc
    if best_auroc < auroc:
        best_auroc = auroc
        model.save(sess, '{}/ckpt'.format(save_path))
    return auroc, prec, rec, f1


with tf.Session() as sess:
    model = BiWGAN(count_list, method, weight=weight, degree=degree)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # model.restore(sess, '{}/ckpt'.format(save_path))
    print('Total params: ', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

    start_time = time.time()
    auroc, prec, rec, f1 = _eval(sess, model, x_val, y_val)
    print('Eval_auc:{:.4f} | prec:{:.4f} | rec:{:.4f} | f1:{:.4f}\tTime Cost:{:.4f}'.format(auroc, prec,
                                                                                            rec, f1,
                                                                                            time.time()-start_time))
    sys.stdout.flush()

    start_time = time.time()
    for i in range(nb_epochs):
        print('==== Training epoch {} ===='.format(i))
        sys.stdout.flush()

        x, y = _shuffle(x, y)
        loss_dis_sum, loss_gen_sum, loss_enc_sum = 0., 0., 0.
        for j, batch_data in DataInput(x, train_batch_size):
            # train D
            if (j % d_g_iter != 0) or (j == 0):
                loss_dis, loss_gen, loss_enc = model.train(sess, batch_data, learning_rate, train_d=True)
            # train G, E
            else:
                loss_dis, loss_gen, loss_enc = model.train(sess, batch_data, learning_rate, train_d=False)
            loss_dis_sum += loss_dis
            loss_gen_sum += loss_gen
            loss_enc_sum += loss_enc

            print_iter = 750 if best_auroc > 0.85 else 1500
            if j % print_iter == 0:
                print('== Epoch {}  Batch {}\tLoss_dis:{:.4f}  Loss_gen:{:.4f}  loss_enc:{:.4f} =='\
                      .format(i, j,
                              loss_dis_sum / print_iter,
                              loss_gen_sum / print_iter,
                              loss_enc_sum / print_iter))
                sys.stdout.flush()
                loss_dis_sum, loss_gen_sum, loss_enc_sum = 0., 0., 0.

                # Validation
                auroc, prec, rec, f1 = _eval(sess, model, x_val, y_val)
                print('Eval_auc:{:.4f} | prec:{:.4f} | rec:{:.4f} | f1:{:.4f}\tBest_auc:{:.4f}'.format(
                    auroc, prec, rec, f1, best_auroc))

        print('Epoch {} Done\tCost time:{:.4f}'.format(i, time.time() - start_time))
        sys.stdout.flush()

    print('Training Done. Best auc:', best_auroc)
    sys.stdout.flush()
