# -*- coding:utf-8 -*-
"""
@file name  : 00_onnx_graph.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-06-01
@brief      : 生成第一个onnx文件
"""

from onnx import TensorProto
from onnx.helper import  make_model, make_node, make_graph, make_tensor_value_info

# 'X' is the name, TensorProto.FLOAT the type, [None, None] the shape
X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])

node1 = make_node('MatMul', ['X', 'A'], ['XA'])
node2 = make_node('Add', ['XA', 'B'], ['Y'])

graph = make_graph([node1, node2],  # nodes
                    'lr',  # a name
                    [X, A, B],  # inputs
                    [Y])  # outputs

onnx_model = make_model(graph)

with open("linear_regression.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

