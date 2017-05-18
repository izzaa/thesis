import numpy as np

from config import Config
from data import BALANCE_WEIGHTS
from layers import *

cnf = {
    'name': __name__.split('.')[-1],
    'w': 112,
    'h': 112,
    'train_dir': 'data/train_tiny',
    'test_dir': 'data/test_tiny',
    'batch_size_train': 128,
    'batch_size_test': 128,
    'balance_weights': np.array(BALANCE_WEIGHTS),
    'balance_ratio': 0.975,
    'final_balance_weights':  np.array([1, 2, 2, 2, 2], dtype=float),
    'aug_params': {
        'zoom_range': (1 / 1.15, 1.15),
        'rotation_range': (0, 360),
        'shear_range': (0, 0),
        'translation_range': (-40, 40),
        'do_flip': True,
        'allow_stretch': True,
    },
    'weight_decay': 0.0005,
    'sigma': 0.5,
    'schedule': {
        0: 0.003,
        150: 0.0003,
        201: 'stop',
    },
}

n = 32
index = 9

layers = [
    (InputLayer, {'shape': (None, 3, cnf['w'], cnf['h'])}),
    (Conv2DLayer, conv_params(n, name='Conv2DLayer_1', filter_size=(5, 5), stride=(2, 2))),
    (Conv2DLayer, conv_params(n, name="Conv2DLayer_2")),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(2 * n, name="Conv2DLayer_3", filter_size=(5, 5), stride=(2, 2))),
    (Conv2DLayer, conv_params(2 * n, name="Conv2DLayer_4")),
    (Conv2DLayer, conv_params(2 * n, name="Conv2DLayer_5")),
    (ConcatLayer, {'incomings':["Conv2DLayer_3", "Conv2DLayer_5"]}),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(4 * n, name="Conv2DLayer_6")),
    (Conv2DLayer, conv_params(4 * n, name="Conv2DLayer_7")),
    (Conv2DLayer, conv_params(4 * n, name="Conv2DLayer_8")),

    (Conv2DLayer, conv_params(2 * n, name="Conv2DLayer" + str(index), filter_size=(1, 1))),

    (Conv2DLayer, conv_params(2 * n, name="Conv2DLayer_3" + str(index + 1), filter_size=(1, 1))),
    (Conv2DLayer, conv_params(2 * n, name="Conv2DLayer_3" + str(index + 2), filter_size=(3, 3))),

    (Conv2DLayer, conv_params(2 * n, name="Conv2DLayer_3" + str(index + 3), filter_size=(1, 1))),
    (Conv2DLayer, conv_params(2 * n, name="Conv2DLayer_3" + str(index + 4), filter_size=(5, 5))),

    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(2 * n, name="Conv2DLayer_3" + str(index + 5), filter_size=(1, 1))),

    (ConcatLayer, {'incomings':["Conv2DLayer" + str(index), "Conv2DLayer" + str(index + 2), "Conv2DLayer" + str(index + 4), "Conv2DLayer" + str(index + 5)]}),

    (ConcatLayer, {'incomings':["Conv2DLayer_6", "Conv2DLayer_8"]}),
    (RMSPoolLayer, pool_params(stride=(3, 3))),
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, dense_params(1024)),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, dense_params(1024)),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DenseLayer, {'num_units': 1}),
]

config = Config(layers=layers, cnf=cnf)
