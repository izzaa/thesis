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

n = 64

layers = [
    (InputLayer, {'shape': (None, 3, cnf['w'], cnf['h'])}),
    (Conv2DLayer, conv_params("Conv2DLayer_1", n, filter_size=(7, 7), stride=(2, 2), pad=3)),
    # (BatchNormLayer, {'incoming':'Conv2DLayer_1', 'name':'BatchNormLayer_2'}),
    # activation or scale conv
    (MaxPool2DLayer, pool_params(name='pool_3')),

# ===============================================================================================================

    # left side 
    (Conv2DLayer, conv_params("Conv2DLayer_4_a", n, filter_size=(1, 1), stride=(1, 1), pad=0)),#, incoming='pool_3')),
    # (BatchNormLayer, {'name':'BatchNormLayer_4_a'}),   
    # activation or scale conv 

    # right side
    (Conv2DLayer, conv_params("Conv2DLayer_4_b", n, filter_size=(1, 1), stride=(1, 1), pad=0)),#, incoming='pool_3')),
    # (BatchNormLayer, {'name':'BatchNormLayer_4_b'}),    
    # activation or scale conv
    (Conv2DLayer, conv_params("Conv2DLayer_5_b", n, filter_size=(3, 3), stride=(1, 1), pad=0)),
    # (BatchNormLayer, {'name':'BatchNormLayer_5_b'}),   
    # activation or scale conv 
    (Conv2DLayer, conv_params("Conv2DLayer_6_b", n, filter_size=(1, 1), stride=(1, 1), pad=0)),
    # (BatchNormLayer, {'name':'BatchNormLayer_6_b'}),  
    # activation or scale conv  

    (ConcatLayer, {'name':'concat_7', 'incomings':["Conv2DLayer_4_a", "Conv2DLayer_6_b"]}),

# ===============================================================================================================
    # left side 

    # right side
    (Conv2DLayer, conv_params("Conv2DLayer_7_b", n, filter_size=(1, 1), stride=(1, 1), pad=0)),#, incoming='concat_7')),
    # (BatchNormLayer, {'name':'BatchNormLayer_7_b'}),    
    # activation or scale conv
    (Conv2DLayer, conv_params("Conv2DLayer_8_b", n, filter_size=(3, 3), stride=(1, 1), pad=0)),
    # (BatchNormLayer, {'name':'BatchNormLayer_8_b'}),   
    # activation or scale conv 
    (Conv2DLayer, conv_params("Conv2DLayer_9_b", n, filter_size=(1, 1), stride=(1, 1), pad=0)),
    # (BatchNormLayer, {'name':'BatchNormLayer_9_b'}),  
    # activation or scale conv  

    (ConcatLayer, {'name':'concat_10', 'incomings':["concat_7", "Conv2DLayer_9_b"]}),

# ===============================================================================================================

    # left side 

    # right side
    (Conv2DLayer, conv_params("Conv2DLayer_11_b", n, filter_size=(1, 1), stride=(1, 1), pad=0)),#, incoming='concat_10')),
    # (BatchNormLayer, {'name':'BatchNormLayer_11_b'}),    
    # activation or scale conv
    (Conv2DLayer, conv_params("Conv2DLayer_12_b", n, filter_size=(3, 3), stride=(1, 1), pad=0)),
    # (BatchNormLayer, {'name':'BatchNormLayer_12_b'}),   
    # activation or scale conv 
    (Conv2DLayer, conv_params("Conv2DLayer_13_b", n, filter_size=(1, 1), stride=(1, 1), pad=0)),
    # (BatchNormLayer, {'name':'BatchNormLayer_13_b'}),  
    # activation or scale conv  

    (ConcatLayer, {'name':'concat_14', 'incomings':["concat_10", "Conv2DLayer_13_b"]}),

# ===============================================================================================================

    # left side 
    (Conv2DLayer, conv_params("Conv2DLayer_15_a", n, filter_size=(1, 1), stride=(1, 1), pad=0)),#, incoming='concat_14')),
    # (BatchNormLayer, {'name':'BatchNormLayer_15_a'}),   

    # right side
    (Conv2DLayer, conv_params("Conv2DLayer_15_b", n, filter_size=(1, 1), stride=(1, 1), pad=0)),#, incoming='concat_14')),
    # (BatchNormLayer, {'name':'BatchNormLayer_15_b'}),    
    # activation or scale conv
    (Conv2DLayer, conv_params("Conv2DLayer_16_b", n, filter_size=(3, 3), stride=(1, 1), pad=0)),
    # (BatchNormLayer, {'name':'BatchNormLayer_16_b'}),   
    # activation or scale conv 
    (Conv2DLayer, conv_params("Conv2DLayer_17_b", n, filter_size=(1, 1), stride=(1, 1), pad=0)),
    # (BatchNormLayer, {'name':'BatchNormLayer_17_b'}),  
    # activation or scale conv  

    (ConcatLayer, {'name':'concat_18', 'incomings':["Conv2DLayer_15_a", "Conv2DLayer_17_b"]}),

# ===============================================================================================================

    # left side 

    # right side
    (Conv2DLayer, conv_params("Conv2DLayer_19_b", n, filter_size=(1, 1), stride=(1, 1), pad=0)),#, incoming='concat_18')),
    # (BatchNormLayer, {'name':'BatchNormLayer_19_b'}),    
    # activation or scale conv
    (Conv2DLayer, conv_params("Conv2DLayer_20_b", n, filter_size=(3, 3), stride=(1, 1), pad=0)),
    # (BatchNormLayer, {'name':'BatchNormLayer_20_b'}),   
    # activation or scale conv 
    (Conv2DLayer, conv_params("Conv2DLayer_21_b", n, filter_size=(1, 1), stride=(1, 1), pad=0)),
    # (BatchNormLayer, {'name':'BatchNormLayer_21_b'}),  
    # activation or scale conv  

    (ConcatLayer, {'name':'concat_22', 'incomings':["concat_18", "Conv2DLayer_21_b"]}),

# ===============================================================================================================

    # left side 

    # right side
    (Conv2DLayer, conv_params("Conv2DLayer_23_b", n, filter_size=(1, 1), stride=(1, 1), pad=0)),#, incoming='concat_22')),
    # (BatchNormLayer, {'name':'BatchNormLayer_23_b'}),    
    # activation or scale conv
    (Conv2DLayer, conv_params("Conv2DLayer_24_b", n, filter_size=(3, 3), stride=(1, 1), pad=0)),
    # (BatchNormLayer, {'name':'BatchNormLayer_24_b'}),   
    # activation or scale conv 
    (Conv2DLayer, conv_params("Conv2DLayer_25_b", n, filter_size=(1, 1), stride=(1, 1), pad=0)),
    # (BatchNormLayer, {'name':'BatchNormLayer_25_b'}),  
    # activation or scale conv  

    (ConcatLayer, {'name':'concat_26', 'incomings':["concat_22", "Conv2DLayer_25_b"]}),

# ===============================================================================================================

    # left side 

    # right side
    (Conv2DLayer, conv_params("Conv2DLayer_27_b", n, filter_size=(1, 1), stride=(1, 1), pad=0)),#, incoming='concat_26')),
    # (BatchNormLayer, {'name':'BatchNormLayer_27_b'}),    
    # activation or scale conv
    (Conv2DLayer, conv_params("Conv2DLayer_28_b", n, filter_size=(3, 3), stride=(1, 1), pad=0)),
    # (BatchNormLayer, {'name':'BatchNormLayer_28_b'}),   
    # activation or scale conv 
    (Conv2DLayer, conv_params("Conv2DLayer_29_b", n, filter_size=(1, 1), stride=(1, 1), pad=0)),
    # (BatchNormLayer, {'name':'BatchNormLayer_29_b'}),  
    # activation or scale conv  

    (ConcatLayer, {'name':'concat_30', 'incomings':["concat_26", "Conv2DLayer_29_b"]}),

# ===============================================================================================================

    # left side 
    (Conv2DLayer, conv_params("Conv2DLayer_31_a", n, filter_size=(1, 1), stride=(1, 1), pad=0)),#, incoming='concat_30')),
    # (BatchNormLayer, {'name':'BatchNormLayer_31_a'}),   

    # right side
    (Conv2DLayer, conv_params("Conv2DLayer_31_b", n, filter_size=(1, 1), stride=(1, 1), pad=0)),#, incoming='concat_30')),
    # (BatchNormLayer, {'name':'BatchNormLayer_31_b'}),    
    # activation or scale conv
    (Conv2DLayer, conv_params("Conv2DLayer_32_b", n, filter_size=(3, 3), stride=(1, 1), pad=0)),
    # (BatchNormLayer, {'name':'BatchNormLayer_32_b'}),   
    # activation or scale conv 
    (Conv2DLayer, conv_params("Conv2DLayer_33_b", n, filter_size=(1, 1), stride=(1, 1), pad=0)),
    # (BatchNormLayer, {'name':'BatchNormLayer_33_b'}),  
    # activation or scale conv  

    (ConcatLayer, {'name':'concat_34', 'incomings':["Conv2DLayer_31_a", "Conv2DLayer_33_b"]}),

# ===============================================================================================================

    # left side 

    # right side
    (Conv2DLayer, conv_params("Conv2DLayer_35_b", n, filter_size=(1, 1), stride=(1, 1), pad=0)),#, incoming='concat_34')),
    # (BatchNormLayer, {'name':'BatchNormLayer_35_b'}),    
    # activation or scale conv
    (Conv2DLayer, conv_params("Conv2DLayer_36_b", n, filter_size=(3, 3), stride=(1, 1), pad=0)),
    # (BatchNormLayer, {'name':'BatchNormLayer_36_b'}),   
    # activation or scale conv 
    (Conv2DLayer, conv_params("Conv2DLayer_37_b", n, filter_size=(1, 1), stride=(1, 1), pad=0)),
    # (BatchNormLayer, {'name':'BatchNormLayer_37_b'}),  
    # activation or scale conv  

    (ConcatLayer, {'name':'concat_38', 'incomings':["concat_34", "Conv2DLayer_37_b"]}),

# ===============================================================================================================

    # left side 

    # right side
    (Conv2DLayer, conv_params("Conv2DLayer_39_b", n, filter_size=(1, 1), stride=(1, 1), pad=0)),#, incoming='concat_38')),
    # (BatchNormLayer, {'name':'BatchNormLayer_39_b'}),    
    # activation or scale conv
    (Conv2DLayer, conv_params("Conv2DLayer_40_b", n, filter_size=(3, 3), stride=(1, 1), pad=0)),
    # (BatchNormLayer, {'name':'BatchNormLayer_40_b'}),   
    # activation or scale conv 
    (Conv2DLayer, conv_params("Conv2DLayer_41_b", n, filter_size=(1, 1), stride=(1, 1), pad=0)),
    # (BatchNormLayer, {'name':'BatchNormLayer_41_b'}),  
    # activation or scale conv  

    (ConcatLayer, {'name':'concat_42', 'incomings':["concat_38", "Conv2DLayer_41_b"]}),

# ===============================================================================================================

    # left side 
    (Conv2DLayer, conv_params("Conv2DLayer_43_a", n, filter_size=(1, 1), stride=(1, 1), pad=0)),#, incoming='concat_42')),
    # (BatchNormLayer, {'name':'BatchNormLayer_43_a'}),   

    # right side
    (Conv2DLayer, conv_params("Conv2DLayer_43_b", n, filter_size=(1, 1), stride=(1, 1), pad=0)),#, incoming='concat_42')),
    # (BatchNormLayer, {'name':'BatchNormLayer_43_b'}),    
    # activation or scale conv
    (Conv2DLayer, conv_params("Conv2DLayer_44_b", n, filter_size=(3, 3), stride=(1, 1), pad=0)),
    # (BatchNormLayer, {'name':'BatchNormLayer_44_b'}),   
    # activation or scale conv 
    (Conv2DLayer, conv_params("Conv2DLayer_45_b", n, filter_size=(1, 1), stride=(1, 1), pad=0)),
    # (BatchNormLayer, {'name':'BatchNormLayer_45_b'}),  
    # activation or scale conv  

    (ConcatLayer, {'name':'concat_46', 'incomings':["Conv2DLayer_43_a", "Conv2DLayer_45_b"]}),

# ===============================================================================================================

    # left side 

    # right side
    (Conv2DLayer, conv_params("Conv2DLayer_47_b", n, filter_size=(1, 1), stride=(1, 1), pad=0)),#, incoming='concat_46')),
    # (BatchNormLayer, {'name':'BatchNormLayer_47_b'}),    
    # activation or scale conv
    (Conv2DLayer, conv_params("Conv2DLayer_48_b", n, filter_size=(3, 3), stride=(1, 1), pad=0)),
    # (BatchNormLayer, {'name':'BatchNormLayer_48_b'}),   
    # activation or scale conv 
    (Conv2DLayer, conv_params("Conv2DLayer_49_b", n, filter_size=(1, 1), stride=(1, 1), pad=0)),
    # (BatchNormLayer, {'name':'BatchNormLayer_49_b'}),  
    # activation or scale conv  

    (ConcatLayer, {'name':'concat_50', 'incomings':["concat_46", "Conv2DLayer_49_b"]}),

# ===============================================================================================================

    # left side 

    # right side
    (Conv2DLayer, conv_params("Conv2DLayer_51_b", n, filter_size=(1, 1), stride=(1, 1), pad=0)),#, incoming='concat_50')),
    # (BatchNormLayer, {'name':'BatchNormLayer_51_b'}),    
    # activation or scale conv
    (Conv2DLayer, conv_params("Conv2DLayer_52_b", n, filter_size=(3, 3), stride=(1, 1), pad=0)),
    # (BatchNormLayer, {'name':'BatchNormLayer_52_b'}),   
    # activation or scale conv 
    (Conv2DLayer, conv_params("Conv2DLayer_53_b", n, filter_size=(1, 1), stride=(1, 1), pad=0)),
    # (BatchNormLayer, {'name':'BatchNormLayer_53_b'}),  
    # activation or scale conv  

    (ConcatLayer, {'name':'concat_54', 'incomings':["concat_50", "Conv2DLayer_53_b"]}),

# ===============================================================================================================

    (RMSPoolLayer, pool_params(stride=(3, 3))),
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, dense_params(1000)),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, dense_params(1024)),
    (FeaturePoolLayer, {'pool_size': 2}),
    (DenseLayer, {'num_units': 1}),
]

config = Config(layers=layers, cnf=cnf)
