import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils import _split_dataset


def _encode_text_dummy(df, name):
    """one-hot encoding"""
    dummies = pd.get_dummies(df.loc[:, name])
    for x in dummies.columns:
        dummy_name = '{}-{}'.format(name, x)
        df.loc[:, dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)


def build_dataset(mode):

    col_names = ["duration", "protocol_type", "service", "flag", "src_bytes",
                 "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                 "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                 "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                 "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                 "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                 "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                 "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                 "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                 "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]

    df = pd.read_csv('data/kddcup.data_10_percent_corrected', header=None, names=col_names)

    # One-hot encoding
    text_l = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']
    for name in text_l:
        _encode_text_dummy(df, name)

    # Label mapping
    labels = df['label'].copy()
    f = lambda x: 1 if (x == 'normal.') else 0
    df.loc[:, 'label'] = labels.map(f)

    df_train = df.sample(frac=0.5, random_state=42)
    df_test = df.loc[~df.index.isin(df_train.index)].copy()

    y_train = df_train['label'].values
    df_train.drop(['label'], axis=1, inplace=True)
    x_train = df_train.values.astype(np.float32)

    y_test = df_test['label'].values
    df_test.drop(['label'], axis=1, inplace=True)
    x_test = df_test.values.astype(np.float32)

    print('raw training set', x_train.shape)

    # Scaler
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # val split
    (x_val, y_val), _ = _split_dataset(x_test, y_test, percentage=0.2)
    print('validation set', x_val.shape)

    # Only use majority class to train
    x_train = x_train[y_train != 1]
    y_train = y_train[y_train != 1]

    print('training set', x_train.shape)
    print('test set', x_test.shape)

    train_set = (x_train, y_train)
    val_set = (x_val, y_val)
    test_set = (x_test, y_test)

    with open('{}_dataset.pkl'.format(mode), 'wb') as f:
        pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(val_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    build_dataset(mode='kdd')


