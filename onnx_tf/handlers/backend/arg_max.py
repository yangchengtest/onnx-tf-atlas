import tensorflow as tf
import copy

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("ArgMax")
@tf_func(tf.argmax)
class ArgMax(BackendHandler):

  @classmethod
  def get_attrs_processor_param(cls):
    return {"default": {"axis": 0}}

  @classmethod
  def _common(cls, node, **kwargs):
    axis = node.attrs.get("axis", 0)
    keepdims = node.attrs.get("keepdims", 1)
    input_format =kwargs.get("input_format", "NCHW")
    input_dict = kwargs["tensor_dict"]
    x = input_dict[node.inputs[0]]
    x_rank = len(x.get_shape())
    attrs = copy.deepcopy(node.attrs)
    if input_format =="NCHW":
      attrs["axis"] = 1
    else:
      attrs["axis"] = x_rank-1
    arg_max = cls.make_tensor_from_onnx_node(node, attrs=attrs,**kwargs)
    if axis==1 and input_format=="NHWC":
      axis = x_rank-1
    if keepdims == 1:
      return [tf.expand_dims(arg_max, axis=axis)]
    return [arg_max]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls._common(node, **kwargs)
