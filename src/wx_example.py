import numpy as np
from keras.utils import to_categorical
from wx_hyperparam import WxHyperParameter
from wx_core import WxSlp


def GetSampleData():
    train_num = 100
    test_num = 100
    input_dim = 2000
    num_cls = 2

    x_train = np.random.random((train_num, input_dim))
    y_train = to_categorical(np.random.randint(num_cls, size=(train_num, 1)), num_classes=num_cls)

    x_test = np.random.random((test_num, input_dim))
    y_test = to_categorical(np.random.randint(num_cls, size=(test_num, 1)), num_classes=num_cls)

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    x_train, y_train, x_val, y_val = GetSampleData()

    hp = WxHyperParameter(epochs=30, learning_ratio=0.01, batch_size=10)
    sel_idx, sel_weight, val_acc = WxSlp(x_train, y_train, x_train, y_train, n_selection=50, hyper_param=hp)

    print ('\nSingle Layer WX')
    print ('selected feature index:',sel_idx)
    print ('selected feature weight:',sel_weight)
    print ('evaluation accuracy:',val_acc)