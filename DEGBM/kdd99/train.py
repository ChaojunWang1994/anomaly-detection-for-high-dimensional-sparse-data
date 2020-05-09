import os
import time
import pickle
import numpy as np
import tensorflow as tf
import sys
from build_dataset import build_dataset
from model import BiWGAN
from utils import DataInput, calc_metric, calc_auc, create_logdir, _shuffle


# hyper parameters
base_dir = os.path.dirname(os.path.realpath(__file__))
mode = 'kdd'
train_batch_size = 64
test_batch_size = 1024
nb_epochs = 200
print_iter = 50
d_g_iter = 6  # (D's training epochs / G's training epochs = d_g_iter - 1)
learning_rate = 1e-4
input_dim = 121
method = 'cross-e'
weight = 0.9
degree = 'euclidean'
logdir = create_logdir(mode, method, weight, degree)
save_path = os.path.join(base_dir, logdir)
best_auprc = 0.
best_f1 = 0.


if not os.path.exists(save_path):
    os.makedirs(save_path)

if not os.path.exists('{}_dataset.pkl'.format(mode)):
    build_dataset(mode)

with open('{}_dataset.pkl'.format(mode), 'rb') as f:
    train_set = pickle.load(f)
    val_set = pickle.load(f)
    test_set = pickle.load(f)

x_train, y_train = train_set
x_val, y_val = val_set


def _eval(sess, model, test_data, label):
    ano_scores = []
    for _, batch_test_data in DataInput(test_data, test_batch_size):
        _ano_score, _, _ = model.eval(sess, batch_test_data)
        # Extend
        ano_scores += list(_ano_score)
    ano_scores = np.array(ano_scores).reshape((-1, 1))

    # Highest 80% are anomalous
    prec, rec, f1 = calc_metric(label, ano_scores)

    # Calculate auprc
    _auprc = calc_auc(label, ano_scores)

    global best_f1
    if best_f1 < f1:
        best_f1 = f1
        model.save(sess, '{}/ckpt'.format(save_path))

    global best_auprc
    if best_auprc < _auprc:
        best_auprc = _auprc

    return prec, rec, f1, _auprc


with tf.Session() as sess:
    model = BiWGAN(input_dim, method, weight, degree)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    prec, rec, f1, auprc = _eval(sess, model, x_val, y_val)
    print('Prec:{:.4f}  |  Rec:{:.4f}  |  F1:{:.4f}  |  Eval_auprc:{:.4f}'.format(prec, rec, f1, auprc))
    sys.stdout.flush()

    # Start training
    start_time = time.time()
    for i in range(nb_epochs):
        print('==== Training epoch {} ===='.format(i))
        sys.stdout.flush()

        # shuffle for each epoch
        x_train, y_train = _shuffle(x_train, y_train)

        loss_dis_sum, loss_gen_sum, loss_enc_sum = 0., 0., 0.
        for j, batch_data in DataInput(x_train, train_batch_size):
            # Update discriminator
            if (j % d_g_iter != 0) or (j == 0):
                loss_dis, loss_gen, loss_enc = model.train(sess, batch_data, learning_rate, train_d=True)

            # Update generator and encoder
            else:
                loss_dis, loss_gen, loss_enc = model.train(sess, batch_data, learning_rate, train_d=False)

            loss_dis_sum += loss_dis
            loss_gen_sum += loss_gen
            loss_enc_sum += loss_enc

            if j % print_iter == 0:
                print('Epoch {} Batch {}\tloss_dis{:.4f}\tloss_gen{:.4f}\tloss_enc{:.4f}'\
                      .format(i,
                              j,
                              loss_dis_sum/print_iter,
                              loss_gen_sum/print_iter,
                              loss_enc_sum/print_iter))
                sys.stdout.flush()

                loss_dis_sum, loss_gen_sum, loss_enc_sum = 0., 0., 0.

                # Validation
                prec, rec, f1, auprc = _eval(sess, model, x_val, y_val)
                print('Prec:{:.4f}  |  Rec:{:.4f}  |  F1:{:.4f}  |  Eval_auprc:{:.4f}'.format(prec, rec, f1, auprc))
                sys.stdout.flush()

                print('Best_f1:{:.4f}\tBest_auprc:{:.4f}\n'.format(best_f1, best_auprc))
                sys.stdout.flush()

        print('Epoch {} Done\tCost time: {:.4f}'.format(i, time.time() - start_time))
        sys.stdout.flush()

    print('Training Done\tBest_auprc:', best_auprc)
    sys.stdout.flush()