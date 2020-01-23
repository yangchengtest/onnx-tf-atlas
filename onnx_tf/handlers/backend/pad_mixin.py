import numpy as np
import tensorflow as tf


class PadMixin(object):

  @classmethod
  def get_padding_as_op(cls, x, pads, input_format='NCHW'):
    num_dim = int(len(pads) / 2)

    tf_pads = np.transpose(np.array(pads).reshape([2, num_dim]))
    if input_format=='NCHW':
      tf_pads = [0, 0, 0, 0] + tf_pads.flatten().tolist()
    else:
      tf_pads = [0, 0]+tf_pads.flatten().tolist()+[0,0]
    print (tf_pads)
    padding = tf.constant(
        np.array(tf_pads).reshape([num_dim + 2, 2])
        .astype(np.int32))  # tf requires int32 paddings
    return tf.pad(x, padding)
