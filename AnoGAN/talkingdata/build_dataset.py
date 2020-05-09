import numpy as np
import pandas as pd
import pickle
import gc
from sklearn.preprocessing import LabelEncoder
from utils import _split_dataset


def _use_cols():
    return ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']


def build_dataset(mode):
    col_names = _use_cols()
    df = pd.read_csv('data/sub_normals_1percent.csv', usecols=col_names, parse_dates=['click_time'])  # 685269
    len_df = len(df)
    df_ano = pd.read_csv('data/anomalies.csv', usecols=col_names, parse_dates=['click_time'])   # 456846
    df = df.append(df_ano)
    del df_ano
    gc.collect()

    # parsing date time
    print('==== parsing second, minute, hour, day, dayofweek ====')
    df['second'] = df['click_time'].dt.second.astype('uint8')
    df['minute'] = df['click_time'].dt.minute.astype('uint8')
    df['hour'] = df['click_time'].dt.hour.astype('uint8')
    df['day'] = df['click_time'].dt.day.astype('uint8')
    df['dayofweek'] = df['click_time'].dt.dayofweek.astype('uint8')

    print(df['is_attributed'].value_counts())
    label = df['is_attributed'].copy()
    df.drop(['click_time', 'is_attributed'], axis=1, inplace=True)

    # Label encoding
    df = df.apply(LabelEncoder().fit_transform)
    print(df.columns)
    x = df.values.astype(np.int32)
    y = label.values.astype(np.int32)
    count_list = np.max(x, axis=0) + 1
    print('sub dataset', x.shape)

    df_nor, nor_label = x[:len_df], y[:len_df]
    df_ano, ano_label = x[len_df:], y[len_df:]

    # train and test split, use only normal class to train
    (x_test, y_test), (x_train, y_train) = _split_dataset(df_nor, nor_label, percentage=0.5)  # 342635

    # sample out test set with specific anomaly ratio = 0.4
    (ano_test, ano_test_label), (ano_train, ano_train_label) = _split_dataset(df_ano, ano_label,
                                                                              percentage=0.5)  # 228423

    x_train, y_train = np.concatenate((x_train, ano_train), axis=0), np.concatenate((y_train, ano_train_label), axis=0)

    x_test, y_test = np.concatenate((x_test, ano_test), axis=0), np.concatenate((y_test, ano_test_label), axis=0)

    print('test set', x_test.shape)
    print('train set', x_train.shape)
    print(np.sum(y_train), np.sum(y_test))
    print(count_list)

    train_set = (x_train, y_train)
    test_set = (x_test, y_test)

    with open('{}_dataset.pkl'.format(mode), 'wb') as f:
        pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(count_list, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    build_dataset(mode='talkingdata')