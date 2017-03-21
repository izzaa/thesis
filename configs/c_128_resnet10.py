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

def get_basic_layer(prev_layer_name, index, n, stride) :
    basic_layer = [
        (Conv2DLayer, conv_params(n, name="Conv2DLayer_" + str(index))),
        (Conv2DLayer, conv_params(n, name="Conv2DLayer_" + str(index+1), stride=stride)),
        (ConcatLayer, {'name': 'concat_' + str(index+2), 'incomings': [prev_layer_name, "Conv2DLayer_" + str(index+1)]}),
    ]
    return basic_layer, index+3, 'concat_' + str(index+2)

def get_all_layer() :
    layers = [
        (InputLayer, {'shape': (None, 3, cnf['w'], cnf['h'])}),
        (Conv2DLayer, conv_params(n, name="Conv2DLayer_1", filter_size=(7, 7), stride=(2, 2))),
        (MaxPool2DLayer, pool_params(name='pool_2')),
    ]
    prev_layer_name = 'pool_2'
    index = 3
    n_temp = n
    for i in range (0, 5) :
        stride = (1,1)
        if i == 2 or i == 4 :
            n_temp = 2 * n
            stride = (2,2)

        l, index, prev_layer_name = get_basic_layer(prev_layer_name, index, n_temp, stride)
        print("got bulk layer on " + str(index))
        layers.extend(l)

    layers.append((RMSPoolLayer, pool_params(stride=(3, 3))))
    layers.append((DropoutLayer, {'p': 0.5}))
    layers.append((DenseLayer, dense_params(1024)))
    layers.append((FeaturePoolLayer, {'pool_size': 2}))
    layers.append((DropoutLayer, {'p': 0.5}))
    layers.append((DenseLayer, dense_params(1024)))
    layers.append((FeaturePoolLayer, {'pool_size': 2}))
    layers.append((DenseLayer, {'num_units': 1}))

    print("layer length " + str(len(layers)))
    return layers

config = Config(layers=get_all_layer(), cnf=cnf)
