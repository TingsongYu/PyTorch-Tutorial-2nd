from pathlib import Path
import numpy as np
import torch
from ..tools.common import logger

class ModelCheckpoint(object):
    '''
    模型保存，两种模式：
    1. 直接保存最好模型
    2. 按照epoch频率保存模型
    '''
    def __init__(self, checkpoint_dir,
                 monitor,
                 arch,mode='min',
                 epoch_freq=1,
                 best = None,
                 save_best_only = True):
        if isinstance(checkpoint_dir,Path):
            checkpoint_dir = checkpoint_dir
        else:
            checkpoint_dir = Path(checkpoint_dir)
        assert checkpoint_dir.is_dir()
        checkpoint_dir.mkdir(exist_ok=True)
        self.base_path = checkpoint_dir
        self.arch = arch
        self.monitor = monitor
        self.epoch_freq = epoch_freq
        self.save_best_only = save_best_only

        # 计算模式
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf

        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        # 这里主要重新加载模型时候
        #对best重新赋值
        if best:
            self.best = best

        if save_best_only:
            self.model_name = f"BEST_{arch}_MODEL.pth"

    def epoch_step(self, state,current):
        '''
        正常模型
        :param state: 需要保存的信息
        :param current: 当前判断指标
        :return:
        '''
        # 是否保存最好模型
        if self.save_best_only:
            if self.monitor_op(current, self.best):
                logger.info(f"\nEpoch {state['epoch']}: {self.monitor} improved from {self.best:.5f} to {current:.5f}")
                self.best = current
                state['best'] = self.best
                best_path = self.base_path/ self.model_name
                torch.save(state, str(best_path))
        # 每隔几个epoch保存下模型
        else:
            filename = self.base_path / f"EPOCH_{state['epoch']}_{state[self.monitor]}_{self.arch}_MODEL.pth"
            if state['epoch'] % self.epoch_freq == 0:
                logger.info(f"\nEpoch {state['epoch']}: save model to disk.")
                torch.save(state, str(filename))

    def bert_epoch_step(self, state,current):
        '''
        适合bert类型模型，适合pytorch_transformer模块
        :param state:
        :param current:
        :return:
        '''
        model_to_save = state['model']
        if self.save_best_only:
            if self.monitor_op(current, self.best):
                logger.info(f"\nEpoch {state['epoch']}: {self.monitor} improved from {self.best:.5f} to {current:.5f}")
                self.best = current
                state['best'] = self.best
                model_to_save.save_pretrained(str(self.base_path))
                output_config_file = self.base_path / 'configs.json'
                with open(str(output_config_file), 'w') as f:
                    f.write(model_to_save.config.to_json_string())
                state.pop("model")
                torch.save(state,self.base_path / 'checkpoint_info.bin')
        else:
            if state['epoch'] % self.epoch_freq == 0:
                save_path = self.base_path / f"checkpoint-epoch-{state['epoch']}"
                save_path.mkdir(exist_ok=True)
                logger.info(f"\nEpoch {state['epoch']}: save model to disk.")
                model_to_save.save_pretrained(save_path)
                output_config_file = save_path / 'configs.json'
                with open(str(output_config_file), 'w') as f:
                    f.write(model_to_save.config.to_json_string())
                state.pop("model")
                torch.save(state, save_path / 'checkpoint_info.bin')
