import os

import mxnet

from mxnet.contrib.onnx import export_model


import mxnet as mx

from configs import ROOT_DIR

print(mx.context.num_gpus())
from mxnet.runtime import feature_list
print(feature_list())

input_shape = '3,112,112'
input_shape = (1,) + tuple( [int(x) for x in input_shape.split(',')] )
print('input-shape:', input_shape)

import onnx

assert onnx.__version__=='1.2.1'

export_model(sym=os.path.join(ROOT_DIR, 'networks/insightface/model-r100-ii/model-symbol.json'),
             params=os.path.join(ROOT_DIR, 'networks/insightface/model-r100-ii/model-0000.params'),
             input_shape=[input_shape],
             onnx_file_path=os.path.join(ROOT_DIR, 'networks/onnx/model.onnx'),
             verbose=True)

