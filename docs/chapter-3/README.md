# 第三章 PyTorch 数据模块

* [第三章 PyTorch 数据模块](README.md)
  
  * [3.1 Dataset](3.1-dataset.md)
  
  * [3.2 DataLoader](3.2-dataloader.md)
  
  * [3.3 Dataset及常用API](3.3-dataset-useful-api.md)
  
  * [3.4 transforms](3.4-transforms.md)
  
  * [3.5 torchvision 经典dataset学习](3.5-torchvision-dataset.md)
  
    

## 第三章简介

经过前两章的铺垫，本章终于可以讲讲项目代码中重要的模块——数据模块。

数据模块包括哪些内容呢？相信大家多少会有一些感觉，不过最好结合具体任务来剖析数据模块。

我们回顾2.2中的COVID-19分类任务，观察一下数据是如何从硬盘到模型输入的。

我们倒着推，

模型接收的训练数据是

* data：outputs = model(data)
* data来自train_loader： for data, labels in train_loader:
* train_loader 来自 DataLoader与train_data：train_loader = DataLoader(dataset=train_data, batch_size=2)
* train_data 来自 COVID19Dataset：train_data = COVID19Dataset(root_dir=img_dir, txt_path=path_txt_train, transform=transforms_func)
* COVID19Dataset继承于Dataset：COVID19Dataset(Dataset)

至此，知道整个数据处理过程会涉及pytorch的两个核心——<font color=red>**Dataset**， **DataLoader**</font>。

Dataset是一个抽象基类，提供给用户定义自己的数据读取方式，最核心在于**getitem**中间对数据的处理。

**DataLoader是pytorch数据加载的核心**，其中包括多个功能，如打乱数据，采样机制（实现均衡1:1采样），多进程数据加载，组装成Batch形式等丰富的功能。

本章将围绕着它们两个展开介绍pytorch的数据读取、预处理、加载等功能。