import copy
import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("Concat")
@tf_func(tf.concat)
class Concat(BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    inputs = [kwargs["tensor_dict"][inp] for inp in node.inputs]
    input_format = kwargs.get("input_format", "NCHW")
    attrs = copy.deepcopy(node.attrs)
    if input_format =="NCHW":
      attrs["axis"] = 1
    else:
      attrs["axis"] = 3

    return [cls.make_tensor_from_onnx_node(node, inputs=[inputs], attrs=attrs)]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_4(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls._common(node, **kwargs)
