# exp1: 动态batch的推理对比
trtexec --onnx=resnet50_bs_dynamic.onnx --saveEngine=demo.engine --minShapes=input:1x3x224x224 --maxShapes=input:64x3x224x224 --optShapes=input:8x3x224x224 > dynamic_1-64-8.log
trtexec --onnx=resnet50_bs_dynamic.onnx --saveEngine=demo.engine --minShapes=input:1x3x224x224 --maxShapes=input:64x3x224x224 --optShapes=input:16x3x224x224 > dynamic_1-64-16.log
trtexec --onnx=resnet50_bs_dynamic.onnx --saveEngine=demo.engine --minShapes=input:1x3x224x224 --maxShapes=input:64x3x224x224 --optShapes=input:32x3x224x224 > dynamic_1-64-32.log
trtexec --onnx=resnet50_bs_dynamic.onnx --saveEngine=demo.engine --minShapes=input:1x3x224x224 --maxShapes=input:64x3x224x224 --optShapes=input:64x3x224x224 > dynamic_1-64-64.log
trtexec --onnx=resnet50_bs_dynamic.onnx --saveEngine=demo.engine --minShapes=input:1x3x224x224 --maxShapes=input:128x3x224x224 --optShapes=input:128x3x224x224 > dynamic_1-128-128.log

# exp2：fp32, fp16, int8 对比
trtexec --onnx=resnet50_bs_dynamic.onnx --saveEngine=demo.engine --minShapes=input:1x3x224x224 --maxShapes=input:64x3x224x224 --optShapes=input:8x3x224x224 --fp16 > dynamic_1-64-8-fp16.log
trtexec --onnx=resnet50_bs_dynamic.onnx --saveEngine=demo.engine --minShapes=input:1x3x224x224 --maxShapes=input:64x3x224x224 --optShapes=input:16x3x224x224 --fp16 > dynamic_1-64-16-fp16.log
trtexec --onnx=resnet50_bs_dynamic.onnx --saveEngine=demo.engine --minShapes=input:1x3x224x224 --maxShapes=input:64x3x224x224 --optShapes=input:32x3x224x224 --fp16 > dynamic_1-64-32-fp16.log
trtexec --onnx=resnet50_bs_dynamic.onnx --saveEngine=demo.engine --minShapes=input:1x3x224x224 --maxShapes=input:64x3x224x224 --optShapes=input:64x3x224x224 --fp16 > dynamic_1-64-64-fp16.log
trtexec --onnx=resnet50_bs_dynamic.onnx --saveEngine=demo.engine --minShapes=input:1x3x224x224 --maxShapes=input:128x3x224x224 --optShapes=input:128x3x224x224 --fp16 > dynamic_1-128-128-fp16.log

trtexec --onnx=resnet50_bs_dynamic.onnx --saveEngine=demo.engine --minShapes=input:1x3x224x224 --maxShapes=input:64x3x224x224 --optShapes=input:8x3x224x224 --int8 > dynamic_1-64-8-int8.log
trtexec --onnx=resnet50_bs_dynamic.onnx --saveEngine=demo.engine --minShapes=input:1x3x224x224 --maxShapes=input:64x3x224x224 --optShapes=input:16x3x224x224 --int8 > dynamic_1-64-16-int8.log
trtexec --onnx=resnet50_bs_dynamic.onnx --saveEngine=demo.engine --minShapes=input:1x3x224x224 --maxShapes=input:64x3x224x224 --optShapes=input:32x3x224x224 --int8 > dynamic_1-64-32-int8.log
trtexec --onnx=resnet50_bs_dynamic.onnx --saveEngine=demo.engine --minShapes=input:1x3x224x224 --maxShapes=input:64x3x224x224 --optShapes=input:64x3x224x224 --int8 > dynamic_1-64-64-int8.log
trtexec --onnx=resnet50_bs_dynamic.onnx --saveEngine=demo.engine --minShapes=input:1x3x224x224 --maxShapes=input:128x3x224x224 --optShapes=input:128x3x224x224 --int8 > dynamic_1-128-128-int8.log