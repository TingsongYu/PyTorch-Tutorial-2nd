import time
import torch
import numpy as np
from torchvision import models

mode_list = "default reduce-overhead max-autotune".split()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实验一：sin函数

def sin_func(x):
    return torch.sin(x) + torch.cos(x)

run_times = 100000
i_data = torch.tensor(1).to(device)
for mode in mode_list:
    torch.cuda.synchronize()
    time_0 = time.time()
    module_compiled = torch.compile(sin_func, mode=mode)
    torch.cuda.synchronize()
    time_1 = time.time()
    
    # warmup
    sin_func(i_data)
    module_compiled(i_data)
    
    torch.cuda.synchronize()
    time_2 = time.time()
    for i in range(run_times):
        sin_func(i_data)
        
    torch.cuda.synchronize()
    time_3 = time.time()
    for i in range(run_times):
        module_compiled(i_data)
    torch.cuda.synchronize()
    time_4 = time.time()
    
    compile_time = time_1 - time_0
    pre_time = time_3 - time_2
    post_time = time_4 - time_3
    speedup_ratio = (pre_time - post_time)/pre_time
    
    print(f"mode: {mode}, 编译耗时:{compile_time:.2f}，编译前运行耗时:{pre_time:.2f}, 编译后运行耗时:{post_time:.2f}，耗时降低比例:{speedup_ratio:.2%}")
    

# 实验二：resnet18

resnet18 = models.resnet18().to(device)
resnet18.eval()
fake_img = torch.randn(16, 3, 224, 224).to(device)

run_times = 100
with torch.no_grad():
    for mode in mode_list:
        torch.cuda.synchronize()
        time_0 = time.time()
        module_compiled = torch.compile(resnet18, mode=mode)
        torch.cuda.synchronize()
        time_1 = time.time()
        
        # warmup 非常关键！
        resnet18(fake_img)
        module_compiled(fake_img)
        
        #
        torch.cuda.synchronize()
        time_2 = time.time()
        for i in range(run_times):
            resnet18(fake_img)
        
        torch.cuda.synchronize()
        time_3 = time.time()
        for i in range(run_times):
            module_compiled(fake_img)
        
        torch.cuda.synchronize()
        time_4 = time.time()

        compile_time = time_1 - time_0
        pre_time = time_3 - time_2
        post_time = time_4 - time_3
        speedup_ratio = (pre_time - post_time)/pre_time

        print(f"mode: {mode}, 编译耗时:{compile_time:.2f}，编译前运行耗时:{pre_time:.2f}, 编译后运行耗时:{post_time:.2f}，耗时降低比例:{speedup_ratio:.2%}")

# 实验三：BERT

from transformers import BertModel, BertTokenizer
import time

# bert = BertModel.from_pretrained('bert-base-uncased')
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# # 准备一批输入数据
# input_text = "Here is some text to encode"
# inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
# inputs = {k: v.to(device) for k, v in inputs.items()}
# bert.to(device)
# bert.eval()

# run_times = 100
# with torch.no_grad():
#     for mode in mode_list:
        
#         # 编译
#         torch.cuda.synchronize()
#         time_0 = time.time()
#         bert_compiled = torch.compile(bert, mode=mode)
#         torch.cuda.synchronize()
#         time_1 = time.time()
        
#         # warmup 非常关键！
#         bert(**inputs)
#         bert_compiled(**inputs)

#         torch.cuda.synchronize()
#         time_2= time.time()
#         for _ in range(run_times): 
#             _ = bert(**inputs)

#         torch.cuda.synchronize()
#         time_3= time.time()
#         for _ in range(run_times):
#             _ = bert_compiled(**inputs)
        
#         torch.cuda.synchronize()
#         time_4= time.time()
        
#         compile_time = time_1 - time_0
#         pre_time = time_3 - time_2
#         post_time = time_4 - time_3
#         speedup_ratio = (pre_time - post_time)/pre_time
        
        
#         print(f"mode: {mode}, 编译耗时:{compile_time:.2f}，编译前运行耗时:{pre_time:.2f}, 编译后运行耗时:{post_time:.2f}，耗时降低比例:{speedup_ratio:.2%}")



# 实验四 numpy

run_times = 100

def numpy_fn2(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.sum(X[:, :, None] * Y[:, None, :], axis=(-2, -1))

def numpy_fn(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    # Step 1: Normalize the input arrays to have zero mean and unit variance
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    Y_mean, Y_std = Y.mean(axis=0), Y.std(axis=0)
    
    # Avoid division by zero in case of zero standard deviation
    X_std[X_std == 0] = 1
    Y_std[Y_std == 0] = 1
    
    X_normalized = (X - X_mean) / X_std
    Y_normalized = (Y - Y_mean) / Y_std
    
    # Step 2: Perform the tensor product followed by sum over last two dimensions
    intermediate_result = np.sum(X_normalized[:, :, None] * Y_normalized[:, None, :], axis=(-2, -1))
    
    # Step 3: Apply thresholding to clip values outside of [-1, 1]
    intermediate_result = np.clip(intermediate_result, -1, 1)
    
    # Step 4: Apply exponential function for non-linearity
    result = np.exp(intermediate_result)
    
    # Step 5: Add a small regularization term to avoid overfitting
    regularization_term = 0.001 * np.sum(X_normalized ** 2 + Y_normalized ** 2, axis=1)
    result += regularization_term
    
    return result

x = np.random.randn(1024, 64)
y = np.random.randn(1024, 64)

for mode in mode_list:
    torch.cuda.synchronize()
    time_0 = time.time()
    numpy_fn_compiled = torch.compile(numpy_fn, mode=mode)
    torch.cuda.synchronize()
    time_1 = time.time()

    # warmup 非常关键！
    numpy_fn(x, y)
    numpy_fn_compiled(x, y)

    #
    torch.cuda.synchronize()
    time_2 = time.time()
    for i in range(run_times):
        numpy_fn(x, y)

    torch.cuda.synchronize()
    time_3 = time.time()
    for i in range(run_times):
        numpy_fn_compiled(x, y)

    torch.cuda.synchronize()
    time_4 = time.time()

    compile_time = time_1 - time_0
    pre_time = time_3 - time_2
    post_time = time_4 - time_3
    speedup_ratio = (pre_time - post_time)/pre_time

    print(f"mode: {mode}, 编译耗时:{compile_time:.2f}，编译前运行耗时:{pre_time:.2f}, 编译后运行耗时:{post_time:.2f}，耗时降低比例:{speedup_ratio:.2%}")

