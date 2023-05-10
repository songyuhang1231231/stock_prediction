import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras


class PrintHot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


def build_model():
    m1 = keras.models.Sequential()
    m1.add(keras.layers.Dense(128, activation='relu', input_shape=[train_x.shape[1]]))
    m1.add(keras.layers.Dense(128, activation='relu'))
    m1.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.RMSprop()
    m1.compile(optimizer=optimizer, loss='mse',
               metrics=['mae', 'mse'])
    return m1


def plot_show(hist):
    hist1 = hist.history
    hist1['epoch'] = hist.epoch
    fig = plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(hist1['epoch'], hist1['mse'], label='train_mse', color='gray')
    plt.plot(hist1['epoch'], hist1['val_mse'], label='val_mse', color='cyan')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(hist1['epoch'], hist1['mae'], label='train_mae', color='gray')
    plt.plot(hist1['epoch'], hist1['val_mae'], label='val_mae', color='cyan')
    plt.legend()
    plt.tight_layout()
    plt.show()


def estimate_model(model_, test):
    loss, mae, mse = model_.evaluate(test, test_y, verbose=2)
    print(f'测试数据MSE{mse},'
          f'测试数据MAE{mae}')
    result = model_.predict(test).flatten()
    plt.scatter(test_y, result)
    plt.plot([-5, 5], [-5, 5])
    plt.show()


def preprocess_data(num):
    data = gp.copy().loc[num, :]
    zde1 = data.pop('涨跌额')
    data = norm(data)
    data['涨跌额'] = zde1
    data = data.values[np.newaxis, ...]
    result = model.predict(data)
    print('判断为{}, 实际为{}, 题目为{}'.format(result[0, 0], gp.copy().loc[num - 1, ['涨跌额']].values[0],
                                                data[0, :]))


gp = pd.read_csv('./股票日线行情cn.csv', encoding='utf_8_sig')
datasets = gp.copy()
datasets.pop('股票代码')
datasets.pop('交易日期')
ds_x = datasets[1:]
dataset_states = ds_x.describe()
dataset_states.pop('涨跌额')
dataset_states = dataset_states.transpose()
norm = (lambda x: (x - dataset_states['mean']) / dataset_states['std'])
zde = ds_x.pop('涨跌额')
ds_x = norm(ds_x)
ds_x['涨跌额'] = zde.values
ds_y = datasets.pop('涨跌额')[:-1]
train_x, test_x, train_y, test_y = train_test_split(ds_x.values, ds_y.values, test_size=0.2, random_state=1)
with tf.device(tf.test.gpu_device_name()):
    model = build_model()
es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)
EPOCHS = 1000
history = model.fit(train_x, train_y, epochs=EPOCHS, verbose=1, validation_split=0.2,
                    callbacks=[es])
plot_show(history)
