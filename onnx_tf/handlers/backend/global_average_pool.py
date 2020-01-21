import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op


@onnx_op("GlobalAveragePool")
class GlobalAveragePool(BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]   
    ##dims = tf.range(tf.rank(x))
    ##print ("GAP:",dims)
    ##_, dim_window = tf.split(dims, [2, tf.size(dims) - 2])
    ##print ("GAP:",dim_window)
    
    return [tf.reduce_mean(x, axis=[2,3], keep_dims=True)]
