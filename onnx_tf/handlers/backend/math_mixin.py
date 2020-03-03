import copy

from .broadcast_mixin import BroadcastMixin


class BasicMathMixin(BroadcastMixin):
  pass


class ArithmeticMixin(BroadcastMixin):
  pass


class ReductionMixin(BroadcastMixin):

  @classmethod
  def _common(cls, node, **kwargs):
    attrs = copy.deepcopy(node.attrs)
    axis = attrs.pop("axes", None)
    if isinstance(axis, (list, tuple)) and len(axis) == 1:
      axis = axis[0]
    input_format =kwargs.get("input_format", "NCHW")
    input_dict = kwargs["tensor_dict"]
    x = input_dict[node.inputs[0]]
    x_rank = len(x.get_shape())
    if isinstance(axis, (list, tuple)) and len(axis) > 1 and input_format=="NHWC":
      newaxis = []
      for axis_item in axis:
        newaxis.append(axis_item-1)
      axis = newaxis  
    if axis==1 and input_format=="NHWC":
      axis = x_rank-1
    print (axis)  
    attrs["axis"] = axis
    # https://github.com/onnx/onnx/issues/585
    attrs["keepdims"] = attrs.pop("keepdims", 1) == 1
    return [cls.make_tensor_from_onnx_node(node, attrs=attrs, **kwargs)]
