# 特点
支持TENSORFLOW NHWC的数据通道推理，规避华为算子问题。</br>
基于开源：
https://github.com/onnx/onnx-tensorflow

使用方式：</br>
CUDA_VISIBLE_DEVICES=11 python //使用CPU进行TF运算推理。</br>
import onnx</br>
from onnx_tf.backend import prepare</br>
onnx_model = onnx.load("../models/xxx.onnx")</br>
tf_rep = prepare(onnx_model,strict=False,input_format="NHWC")</br>

# 缺点：
1.当前不支持TRANSPOSE，中间层TRANSPOSE数据处理会乱。</br>
2.无法规避华为不支持的算子，如ADDV2等，请自己手动修改。</br>
