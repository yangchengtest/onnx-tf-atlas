import tensorflow as tf
import numpy as np
from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op


@onnx_op("Gemm")
class Gemm(BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    x = tensor_dict[node.inputs[0]]
    x = tf.layers.flatten(x)
    y = tensor_dict[node.inputs[1]]
    z = tensor_dict[node.inputs[2]]
    sess = tf.Session()
    if node.attrs.get("transA", 0):
      sess = tf.Session()
      in_weights_np = None
      with sess.as_default():
        in_weights_np = x.eval().astype(np.float32)
        in_weights_np = np.transpose(in_weights_np,[1,0])
      x = in_weights_np
    if node.attrs.get("transB", 0):
      sess = tf.Session()
      in_weights_np = None
      with sess.as_default():
        in_weights_np = y.eval().astype(np.float32)
        in_weights_np = np.transpose(in_weights_np,[1,0])
      y = in_weights_np
    alpha = node.attrs.get("alpha", 1.0)
    beta = node.attrs.get("beta", 1.0)
    mul = tf.matmul(x, y)
    mul = tf.expand_dims(mul, 1)
    mul = tf.expand_dims(mul, 1)
    return [tf.add(alpha * mul,beta * z)]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_7(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_9(cls, node, **kwargs):
    return cls._common(node, **kwargs)
