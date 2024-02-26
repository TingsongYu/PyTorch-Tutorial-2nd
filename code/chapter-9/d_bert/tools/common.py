import os
import random
import torch
import numpy as np
import json
import pickle
import torch.nn as nn
from collections import OrderedDict
from pathlib import Path
import logging

logger = logging.getLogger()
def print_config(config):
    info = "Running with the following configs:\n"
    for k, v in config.items():
        info += f"\t{k} : {str(v)}\n"
    print("\n" + info + "\n")
    return

def init_logger(log_file=None, log_file_level=logging.NOTSET):
    '''
    Example:
        >>> init_logger(log_file)
        >>> logger.info("abc'")
    '''
    if isinstance(log_file,Path):
        log_file = str(log_file)
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        # file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger

def seed_everything(seed=1029):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def prepare_device(n_gpu_use):
    """
    setup GPU device if available, move model into configured device
    # 如果n_gpu_use为数字，则使用range生成list
    # 如果输入的是一个list，则默认使用list[0]作为controller
     """
    if not n_gpu_use:
        device_type = 'cpu'
    else:
        n_gpu_use = n_gpu_use.split(",")
        device_type = f"cuda:{n_gpu_use[0]}"
    n_gpu = torch.cuda.device_count()
    if len(n_gpu_use) > 0 and n_gpu == 0:
        logger.warning("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        device_type = 'cpu'
    if len(n_gpu_use) > n_gpu:
        msg = f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are available on this machine."
        logger.warning(msg)
        n_gpu_use = range(n_gpu)
    device = torch.device(device_type)
    list_ids = n_gpu_use
    return device, list_ids


def model_device(n_gpu, model):
    '''
    判断环境 cpu还是gpu
    支持单机多卡
    :param n_gpu:
    :param model:
    :return:
    '''
    device, device_ids = prepare_device(n_gpu)
    if len(device_ids) > 1:
        logger.info(f"current {len(device_ids)} GPUs")
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    if len(device_ids) == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_ids[0])
    model = model.to(device)
    return model, device


def restore_checkpoint(resume_path, model=None):
    '''
    加载模型
    :param resume_path:
    :param model:
    :param optimizer:
    :return:
    注意： 如果是加载Bert模型的话，需要调整，不能使用该模式
    可以使用模块自带的Bert_model.from_pretrained(state_dict = your save state_dict)
    '''
    if isinstance(resume_path, Path):
        resume_path = str(resume_path)
    checkpoint = torch.load(resume_path)
    best = checkpoint['best']
    start_epoch = checkpoint['epoch'] + 1
    states = checkpoint['state_dict']
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(states)
    else:
        model.load_state_dict(states)
    return [model,best,start_epoch]


def save_pickle(data, file_path):
    '''
    保存成pickle文件
    :param data:
    :param file_name:
    :param pickle_path:
    :return:
    '''
    if isinstance(file_path, Path):
        file_path = str(file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(input_file):
    '''
    读取pickle文件
    :param pickle_path:
    :param file_name:
    :return:
    '''
    with open(str(input_file), 'rb') as f:
        data = pickle.load(f)
    return data


def save_json(data, file_path):
    '''
    保存成json文件
    :param data:
    :param json_path:
    :param file_name:
    :return:
    '''
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    # if isinstance(data,dict):
    #     data = json.dumps(data)
    with open(str(file_path), 'w') as f:
        json.dump(data, f)

def save_numpy(data, file_path):
    '''
    保存成.npy文件
    :param data:
    :param file_path:
    :return:
    '''
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    np.save(str(file_path),data)

def load_numpy(file_path):
    '''
    加载.npy文件
    :param file_path:
    :return:
    '''
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    np.load(str(file_path))

def load_json(file_path):
    '''
    加载json文件
    :param json_path:
    :param file_name:
    :return:
    '''
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    with open(str(file_path), 'r') as f:
        data = json.load(f)
    return data

def json_to_text(file_path,data):
    '''
    将json list写入text文件中
    :param file_path:
    :param data:
    :return:
    '''
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    with open(str(file_path), 'w') as fw:
        for line in data:
            line = json.dumps(line, ensure_ascii=False)
            fw.write(line + '\n')

def save_model(model, model_path):
    """ 存储不含有显卡信息的state_dict或model
    :param model:
    :param model_name:
    :param only_param:
    :return:
    """
    if isinstance(model_path, Path):
        model_path = str(model_path)
    if isinstance(model, nn.DataParallel):
        model = model.module
    state_dict = model.state_dict()
    for key in state_dict:
        state_dict[key] = state_dict[key].cpu()
    torch.save(state_dict, model_path)

def load_model(model, model_path):
    '''
    加载模型
    :param model:
    :param model_name:
    :param model_path:
    :param only_param:
    :return:
    '''
    if isinstance(model_path, Path):
        model_path = str(model_path)
    logging.info(f"loading model from {str(model_path)} .")
    states = torch.load(model_path)
    state = states['state_dict']
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state)
    return model


class AverageMeter(object):
    '''
    computes and stores the average and current value
    Example:
        >>> loss = AverageMeter()
        >>> for step,batch in enumerate(train_data):
        >>>     pred = self.model(batch)
        >>>     raw_loss = self.metrics(pred,target)
        >>>     loss.update(raw_loss.item(),n = 1)
        >>> cur_loss = loss.avg
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def summary(model, *inputs, batch_size=-1, show_input=True):
    '''
    打印模型结构信息
    :param model:
    :param inputs:
    :param batch_size:
    :param show_input:
    :return:
    Example:
        >>> print("model summary info: ")
        >>> for step,batch in enumerate(train_data):
        >>>     summary(self.model,*batch,show_input=True)
        >>>     break
    '''

    def register_hook(module):
        def hook(module, input, output=None):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = f"{class_name}-{module_idx + 1}"
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size

            if show_input is False and output is not None:
                if isinstance(output, (list, tuple)):
                    for out in output:
                        if isinstance(out, torch.Tensor):
                            summary[m_key]["output_shape"] = [
                                [-1] + list(out.size())[1:]
                            ][0]
                        else:
                            summary[m_key]["output_shape"] = [
                                [-1] + list(out[0].size())[1:]
                            ][0]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == model)):
            if show_input is True:
                hooks.append(module.register_forward_pre_hook(hook))
            else:
                hooks.append(module.register_forward_hook(hook))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)
    model(*inputs)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("-----------------------------------------------------------------------")
    if show_input is True:
        line_new = f"{'Layer (type)':>25}  {'Input Shape':>25} {'Param #':>15}"
    else:
        line_new = f"{'Layer (type)':>25}  {'Output Shape':>25} {'Param #':>15}"
    print(line_new)
    print("=======================================================================")

    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        if show_input is True:
            line_new = "{:>25}  {:>25} {:>15}".format(
                layer,
                str(summary[layer]["input_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
            )
        else:
            line_new = "{:>25}  {:>25} {:>15}".format(
                layer,
                str(summary[layer]["output_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
            )

        total_params += summary[layer]["nb_params"]
        if show_input is True:
            total_output += np.prod(summary[layer]["input_shape"])
        else:
            total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]

        print(line_new)

    print("=======================================================================")
    print(f"Total params: {total_params:0,}")
    print(f"Trainable params: {trainable_params:0,}")
    print(f"Non-trainable params: {(total_params - trainable_params):0,}")
    print("-----------------------------------------------------------------------")