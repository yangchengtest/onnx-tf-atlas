import copy

import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
import numpy as np

@onnx_op("Reshape")
@tf_func(tf.reshape)
class Reshape(BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    tensor = kwargs["tensor_dict"][node.inputs[0]]
    is_const = False
    if cls.SINCE_VERSION == 1:
      shape = tf.constant(node.attrs["shape"], dtype=tf.int64)
    else:  # since_version >= 5
      shape = tf.cast(kwargs["tensor_dict"][node.inputs[1]], tf.int64)
      is_const = True
    input_shape = tf.shape(tensor, out_type=tf.int64)
    need_perm = kwargs.get("reshape_perm",0)
    if need_perm == 1:
      input_dict = kwargs["tensor_dict"]
      x = input_dict[node.inputs[0]]
      #NCHW to NHWC
      x = tf.transpose(x,[0,2,3,1])
      sess = tf.Session()
      with sess.as_default():
        shape_val = shape.eval().astype(np.int64)
        attrs = copy.deepcopy(node.attrs)
        attrs.pop("shape", None)
        y = cls.make_tensor_from_onnx_node(
            node, inputs=[x, shape_val], attrs=attrs, **kwargs)     
        y_shape = len(y.get_shape())
        #NHWC to NCHW
        perm = [0,y_shape-1]
        for index in range(2,y_shape):
          perm.append(index-1)
        print ("perm",perm)            
        z = tf.transpose(y,perm)
      return [z] 
    elif need_perm == 2:
      input_dict = kwargs["tensor_dict"]
      x = input_dict[node.inputs[0]]    
      #NHWC to NCHW
      x = tf.transpose(x,[0,3,1,2])
      sess = tf.Session()
      with sess.as_default():
        shape_val = shape.eval().astype(np.int64)
        attrs = copy.deepcopy(node.attrs)
        attrs.pop("shape", None)
        y = cls.make_tensor_from_onnx_node(
            node, inputs=[x, shape_val], attrs=attrs, **kwargs) 
        y_shape = len(y.get_shape())
        #NCHW to NHWC
        perm = [0]
        for index in range(2,y_shape):
          perm.append(index)
        perm.append(1)  
        print ("perm",perm)   
        z = tf.transpose(y,perm)
      return [z]  
    elif need_perm == 3:    
      input_dict = kwargs["tensor_dict"]
      x = input_dict[node.inputs[0]]   
      sess = tf.Session()
      with sess.as_default():
        shape_val = shape.eval().astype(np.int64) 
        print ("input_shape",shape_val)        
        shape_len = len(list(shape_val))
        shape_result = [1]
        if shape_len == 4:
          shape_result = shape_val
        elif shape_len>=3:
          for i in range(shape_len-2,shape_len):
            shape_result.append(shape_val[i])
          for i in range(1,shape_len-2):
            shape_result.append(shape_val[i])
        else:
          shape_result.append(shape_val[1])
        print("shape",shape_result)  
        attrs = copy.deepcopy(node.attrs)
        attrs.pop("shape", None)
        return [cls.make_tensor_from_onnx_node(
            node, inputs=[x, shape_result], attrs=attrs, **kwargs)]           
    elif is_const:
      sess = tf.Session()
      with sess.as_default():
        shape_val = shape.eval().astype(np.int64)
        indices = list(np.where(shape_val==-1))
        if indices is not None:
          shape_val_first = shape_val[:int(indices[0])]
          shape_val_second = shape_val[int(indices[0])+1:]
          shape_val_end = np.array(-1)
          shape_val = np.hstack((shape_val_first,shape_val_second))
          shape_val = np.hstack((shape_val,shape_val_end))
        print ("shape_val",shape_val)   
        copied_shape = shape_val
    else:
    # Extract indicies of the shape paramter where
    # a copy from the original dimension size is needed.
      copy_indices = tf.squeeze(
          tf.where(tf.equal(shape, tf.constant(0, dtype=tf.int64))), -1)
    
      indices_gathered = tf.gather(input_shape, copy_indices)
      indices_scattered = tf.sparse_to_dense(copy_indices,
                                           tf.cast(tf.shape(shape), tf.int64),
                                           indices_gathered)

    # Perform the copy wherever requested (wherever dim_size == 0)
      copied_shape = shape + indices_scattered
      
    attrs = copy.deepcopy(node.attrs)
    attrs.pop("shape", None)
    return [
        cls.make_tensor_from_onnx_node(
            node, inputs=[tensor, copied_shape], attrs=attrs, **kwargs)
    ]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_5(cls, node, **kwargs):
    return cls._common(node, **kwargs)