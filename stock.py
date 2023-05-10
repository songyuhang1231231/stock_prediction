import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import time
import os
import tushare as ts
import matplotlib.pyplot as plt
from tensorflow import keras

EPOCHS = 1000


class PrintHot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


def preprocess_data(num, model):
    model, gp, norm = model
    data = gp.copy().loc[num, :]
    zde1 = data.pop('change_x')
    data.pop('trade_date')
    data.pop('ts_code_x')
    data.pop('ts_code_y')
    data = norm(data)
    data['change_x'] = zde1
    data = data.values[np.newaxis, ...]
    result = model.predict(data.astype('float32'))
    print('判断为{}, 实际为{},'
          ' 题目为{}'.format(result[0, 0], gp.copy().loc[num - 1, ['change_x']].values[0], data[0, :]))


def preprocess_data_(num, model):
    model, gp, norm = model
    data = gp.copy().loc[num, :]
    zde1 = data.pop('change_x')
    data.pop('trade_date')
    data.pop('ts_code_x')
    data.pop('ts_code_y')
    data = norm(data)
    data['change_x'] = zde1
    data = data.values[np.newaxis, ...]
    result = model.predict(data.astype('float32'))
    print('判断为{},'
          ' 题目为{}'.format(result[0, 0], data[0, :]))


def build_model():
    m1 = keras.models.Sequential()
    m1.add(keras.layers.Dense(32, activation='relu',
                              input_shape=[train_x.shape[1]], kernel_regularizer=keras.regularizers.l2()))
    m1.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(0.001)
    m1.compile(optimizer=optimizer, loss='mse',
               metrics=['mae', 'mse'])
    return m1


def plot_show(hist):
    global path
    hist1 = hist.history
    hist1['epoch'] = hist.epoch
    plt.subplot(1, 2, 1)
    plt.plot(hist1['epoch'], hist1['mse'], label='train_mse', color='gray')
    plt.plot(hist1['epoch'], hist1['val_mse'], label='val_mse', color='cyan')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(hist1['epoch'], hist1['mae'], label='train_mae', color='gray')
    plt.plot(hist1['epoch'], hist1['val_mae'], label='val_mae', color='cyan')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./plt/{path}.png')
    plt.close()


def read_data(path1):
    gp = pd.read_csv(path1, encoding='utf_8_sig')
    datasets = gp.copy()
    datasets.pop('ts_code_x')
    datasets.pop('ts_code_y')
    datasets.pop('trade_date')
    ds_x1 = datasets[1:]
    dataset_states = ds_x1.describe()
    dataset_states.pop('change_x')
    dataset_states = dataset_states.transpose()
    norm = (lambda x: (x - dataset_states['mean']) / dataset_states['std'])
    zde = ds_x1.pop('change_x')
    ds_x1 = norm(ds_x1)
    ds_x1['change_x'] = zde.values
    ds_y1 = datasets.pop('change_x')[:-1]
    return ds_x1.values, ds_y1.values, gp, norm


def get_tushare(ts_code='000153.SZ', start_day='20100101', end_day='20230420'):
    df = ts.pro_bar(ts_code=ts_code, adj='qfq', start_date=start_day, end_date=end_day)
    df1 = ts.pro_bar(ts_code=ts_code, adj='hfq', start_date=start_day, end_date=end_day)
    df = pd.merge(df, df1, on='trade_date')
    df.to_csv('./股票数据/{}.csv'.format(ts_code))
    print('保存成功!')


if not os.path.exists('./plt'):
    os.mkdir('./plt')
token_list = ['c80b73a45affc2ebbf9c19a72a1d9183de2a060a4569209dd58ef32e',
              'c8a66e8744fe78f180baa4925b3e928456e75b9ef92908ea1be6810d',
              '33a4b73f97bd260f3a236c22c24cc5392871bb5e9f82cca6b0a95d1f',
              '5cc777d657c7a4fb9ca7fa9c2b76d09879b01c92661438b0a8fb6b49',
              '25a0a57951c2c60fd190d776c6baa3bdfbfef9dd324f28ebc0ec4230',
              '15784befcea92afbf00ce04480987b0b9918c7d371a254c3e7880990',
              '261254177312f03c178189cf6f019d4ac248df6559f3d4cd09d349c2',
              'af689d2d1177a3cf791617c8714de5067217ad768c65e0697cabbdec',
              '8e7d8f41808064385f0e041d92ca95684767755f1134188a7fdfdd47',
              '6366b436714ea3e40c1cacf4019663b3fec5cce7fe250d74a88ba59a']
token = token_list[-2]
ts.set_token(token)
pro = ts.pro_api(token)


def get_gu_piao(name):
    try:
        get_tushare(name)
    except Exception as e:
        print(e)


def train_model(x1, y1):
    model = build_model()
    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(x1, y1, epochs=EPOCHS, verbose=0, validation_split=0.2,
                        callbacks=[PrintHot(), es])
    return history, model


GP_list = ['300459.SZ', '300182.SZ', '000988.SZ', '002624.SZ', '002354.SZ',
           '300166.SZ', '601360.SH']
path_list = os.listdir('./股票数据')
model_items = []
for path in GP_list[3:]:
    start = time.time()
    if path+'.csv' in path_list:
        ds_x, ds_y, gp1, norm1 = read_data(os.path.join('./股票数据', path+'.csv'))
    else:
        get_tushare(path)
        ds_x, ds_y, gp1, norm1 = read_data(os.path.join('./股票数据', path+'.csv'))
    train_x, test_x, train_y, test_y = train_test_split(ds_x, ds_y, test_size=0.2, random_state=1)
    h, model1 = train_model(train_x, train_y)
    model_items.append([model1, gp1, norm1])
    plot_show(h)
    model1.save(f'{path[:6]}_model')
    print(f'此回合用时{time.time() - start}')
    break
