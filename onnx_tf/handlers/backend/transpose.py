import tensorflow as tf
import copy
from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func


@onnx_op("Transpose")
@tf_func(tf.transpose)
class Transpose(BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    input_dict = kwargs["tensor_dict"]
    x = input_dict[node.inputs[0]] 
    attrs = copy.deepcopy(node.attrs)
    input_format =kwargs.get("input_format", "NCHW")
    if input_format =="NHWC":
      if attrs.get("perm",None) is not None:
        perm = attrs.get("perm",None)
        if len(perm)==4:  
          attrs["perm"] = [0,3,1,2]
        else:  
          need_perm = kwargs.get("reshape_perm",0)
          ##shuffle block:
          if need_perm == 3:
            attrs["perm"] = [0,1,2,4,3]          
        print(attrs)
        return [cls.make_tensor_from_onnx_node(node, attrs=attrs, **kwargs)] 
    return [cls.make_tensor_from_onnx_node(node, **kwargs)]
