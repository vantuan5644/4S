import mxnet as mx
print(mx.context.num_gpus())
from mxnet.runtime import feature_list
print(feature_list())