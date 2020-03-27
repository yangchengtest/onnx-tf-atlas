import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from .math_mixin import ArithmeticMixin


@onnx_op("Mul")
@tf_func(tf.multiply)
class Mul(ArithmeticMixin, BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.limited_broadcast(node, **kwargs)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls.limited_broadcast(node, **kwargs)

  @classmethod
  def version_7(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    x = tensor_dict[node.inputs[0]]
    y = tensor_dict[node.inputs[1]]
    print(y)
    x_rank = len(x.get_shape())
    y_rank = len(y.get_shape())
    print (x_rank,y_rank)
    for i in range(4-x_rank):
      x = tf.expand_dims(x, i)
    for i in range(4-y_rank):
      y = tf.expand_dims(y, i)
      print (y)
    x = tf.cast(x,dtype=tf.float32)      
    y = tf.cast(y,dtype=tf.float32)  
    print (y)    
    return [tf.multiply(x,y)]
