# Copyright (c) 2024 westlake-repl
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
# SPDX-License-Identifier: MIT
# This file has been modified by Junyi Chen.
#
# Original file was released under MIT, with the full license text
# available at https://choosealicense.com/licenses/mit/.
#
# This modified file is released under the same license.

import os
import sys
from logging import getLogger
from time import time
import time as t
import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from tqdm import tqdm
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
import gc
import bitsandbytes as bnb

from ..data.dataset import BatchTextDataset
from ..data.dataset.collate_fn import customize_rmpad_collate
from torch.utils.data import DataLoader
from ..evaluator import Evaluator, Collector
from ..utils import ensure_dir, get_local_time, early_stopping, calculate_valid_score, dict2str, \
    get_tensorboard, set_color, get_gpu_usage, WandbLogger
from ..utils.lr_scheduler import *

import lightning as L
from lightning.fabric.strategies import DeepSpeedStrategy, DDPStrategy


class Trainer(object):
    def __init__(self, config, model):
        super(Trainer, self).__init__()
        self.config = config
        self.model = model
        self.logger = getLogger()

        self.wandblogger = WandbLogger(config)

        self.optim_args = config['optim_args']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        #早停（Early Stopping)
        self.stopping_step = config['stopping_step']
        #防止梯度爆炸，将梯度范数限制在指定范围内
        self.clip_grad_norm = config.get('clip_grad_norm', 1.0)
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.gpu_available = torch.cuda.is_available() and config['use_gpu']
        self.device = config['device']
        # 获取当前进程在分布式训练中的排名
        self.rank = torch.distributed.get_rank()
        #只在rank 0进程初始化TensorBoard
        if self.rank == 0:
            self.tensorboard = get_tensorboard(self.logger)

        self.checkpoint_dir = config['checkpoint_dir']
        if self.rank == 0:
            ensure_dir(self.checkpoint_dir)

        self.saved_model_name = '{}-{}.pth'.format(self.config['model'], 0)
        self.saved_model_file = os.path.join(self.checkpoint_dir, self.saved_model_name)

        self.use_text = config['use_text']

        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.train_loss_dict = dict()
        if config['cpu_optimizer']:
            self.logger.info("use cpu oyimizer")
            self.optimizer = self._build_cpu_optimizer()
        elif config['use_8-bit_optim']:
            self.logger.info("use 8-bit optimizer")
            self.optimizer = self._build_smaller_optimizer()
        else:
            self.optimizer = self._build_optimizer()
        self.update_interval = config['update_interval'] if config['update_interval'] else 20
        self.scheduler_config = config['scheduler_args']

        #根据配置冻结指定前缀的参数
        if config['freeze_prefix'] or config['freeze_ad']:
            freeze_prefix = config['freeze_prefix'] if config['freeze_prefix'] else []
            if config['freeze_ad']:
                freeze_prefix.extend(['item_llm', 'item_emb_tokens'])
            if not config['ft_item']:
                #冻结适配器参数
                freeze_prefix.extend(['item_embedding'])
            #实际执行参数冻结操作，遍历模型参数并冻结匹配前缀的参数
            self._freeze_params(freeze_prefix)

        #记录所有参数的名称、大小和梯度状态
        for n, p in self.model.named_parameters():
            self.logger.info(f"{n} {p.size()} {p.requires_grad}")

        self.eval_collector = Collector(config)
        self.evaluator = Evaluator(config)
        self.item_feature = None
        self.tot_item_num = None

    #冻结指定前缀的模型参数
    def _freeze_params(self, freeze_prefix):
        for name, param in self.model.named_parameters():
            for prefix in freeze_prefix:
                if name.startswith(prefix):
                    self.logger.info(f"freeze_params: {name}")
                    param.requires_grad = False

    #根据配置构建学习率调度器 支持三种调度器：余弦退火、线性衰减、常数学习率
    def _build_scheduler(self, warmup_steps=None, tot_steps=None):
        if self.scheduler_config['type'] == 'cosine':
            self.logger.info(f"Use consine scheduler with {warmup_steps} warmup {tot_steps} total steps")
            return get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, tot_steps)
        elif self.scheduler_config['type'] == 'liner':
            self.logger.info(f"Use linear scheduler with {warmup_steps} warmup {tot_steps} total steps")
            return get_linear_schedule_with_warmup(self.optimizer, warmup_steps, tot_steps)
        else:
            self.logger.info(f"Use constant scheduler")
            return get_constant_schedule(self.optimizer)
    #？

    def _build_cpu_optimizer(self):
        # 必须导入 DeepSpeed 的 CPU 优化器
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        
        # 为了方便后续代码书写，我们将 DeepSpeedCPUAdam 赋值给 OptimizerClass
        # 这样逻辑结构可以和 _build_optimizer 保持完全一致
        OptimizerClass = DeepSpeedCPUAdam

        if len(self.optim_args) == 4:
            params = self.model.named_parameters()
            modal_params = []   # 视觉模态相关参数
            recsys_params = []  # 推荐系统核心参数
            modal_decay_params = []  # 模态衰减参数
            recsys_decay_params = [] # 推荐系统衰减参数
            decay_check_name = self.config['decay_check_name']
            
            for index, (name, param) in enumerate(params):
                if param.requires_grad:
                    # 第一级分类：基于参数名称
                    if 'visual_encoder' in name:
                        modal_params.append(param) 
                    else:
                        recsys_params.append(param) 

                    # 第二级分类：基于衰减检查名称
                    if decay_check_name:
                        if decay_check_name in name:
                            modal_decay_params.append(param) 
                        else:
                            recsys_decay_params.append(param) 

            if decay_check_name:
                # 使用 DeepSpeedCPUAdam
                optimizer = OptimizerClass([
                    {
                        'params': modal_decay_params,
                        'lr': self.optim_args['modal_lr'],
                        'weight_decay': self.optim_args['modal_decay']
                    },
                    {
                        'params': recsys_decay_params,
                        'lr': self.optim_args['rec_lr'],
                        'weight_decay': self.optim_args['rec_decay']
                    }
                ])
                optim_output = set_color(f'recsys_decay_params_len: {len(recsys_decay_params)}  modal_params_decay_len: {len(modal_decay_params)}', 'blue')
                self.logger.info(optim_output)
            else:
                # 使用 DeepSpeedCPUAdam
                optimizer = OptimizerClass([
                    {
                        'params': modal_params,
                        'lr': self.optim_args['modal_lr'],
                        'weight_decay': self.optim_args['modal_decay']
                    },
                    {
                        'params': recsys_params,
                        'lr': self.optim_args['rec_lr'],
                        'weight_decay': self.optim_args['rec_decay']
                    }
                ])
                optim_output = set_color(f'recsys_lr_params_len: {len(recsys_params)}  modal_lr_params_len: {len(modal_params)}', 'blue')
                self.logger.info(optim_output)

        elif self.config['lr_mult_prefix'] and self.config['lr_mult_rate']:
            normal_params_dict = {
                "params": [],
                "lr": self.optim_args['learning_rate'],
                "weight_decay": self.optim_args['weight_decay']
            }
            high_lr_params_dict = {
                "params": [],
                "lr": self.optim_args['learning_rate'] * self.config['lr_mult_rate'],
                "weight_decay": self.optim_args['weight_decay']
            }
            self.logger.info(f'Use higher lr rate {self.config["lr_mult_rate"]} x {self.optim_args["learning_rate"]} for prefix {self.config["lr_mult_prefix"]}')

            for n, p in self.model.named_parameters():
                if any(n.startswith(x) for x in self.config['lr_mult_prefix']):
                    self.logger.info(f"high lr param: {n} {self.optim_args['learning_rate'] * self.config['lr_mult_rate']}")
                    high_lr_params_dict["params"].append(p)
                else:
                    normal_params_dict["params"].append(p)
            
            # 使用 DeepSpeedCPUAdam
            optimizer = OptimizerClass([normal_params_dict, high_lr_params_dict])

        elif self.config['optimizer_kwargs']:
            params = self.model.parameters()
            self.config['optimizer_kwargs']['optimizer']['params']['lr'] = self.optim_args['learning_rate']
            self.config['optimizer_kwargs']['optimizer']['params']['weight_decay'] = self.optim_args['weight_decay']
            
            # 这里本身就是调用 DeepSpeed 的逻辑，直接透传参数
            optimizer = OptimizerClass(params, **self.config['optimizer_kwargs']['optimizer']['params'])
            
        else:
            params = self.model.parameters()
            # 使用 DeepSpeedCPUAdam
            optimizer = OptimizerClass(params, lr=self.optim_args['learning_rate'], weight_decay=self.optim_args['weight_decay'])
            
        return optimizer
    # 根据配置文件（config）中不同的参数设置，灵活地创建不同策略的 PyTorch 优化器
    def _build_optimizer(self):
        if len(self.optim_args) == 4:
            # 处理多模态推荐模型
            params = self.model.named_parameters()
            modal_params = []  # 视觉模态相关参数
            recsys_params = []  # 推荐系统核心参数
            modal_decay_params = []  # 模态衰减参数
            recsys_decay_params = []  # 推荐系统衰减参数
            decay_check_name = self.config['decay_check_name']
            for index, (name, param) in enumerate(params):
                if param.requires_grad:
                    # 第一级分类：基于参数名称
                    if 'visual_encoder' in name:
                        modal_params.append(param)  # 视觉编码器相关参数
                    else:
                        recsys_params.append(param)  # 其他所有参数

                    # 第二级分类：基于衰减检查名称
                    if decay_check_name:
                        if decay_check_name in name:
                            modal_decay_params.append(param)  # 需要特殊衰减的参数
                        else:
                            recsys_decay_params.append(param)  # 普通衰减的参数

            if decay_check_name:
                # 使用细粒度的衰减分组
                optimizer = optim.AdamW([
                    {
                        'params': modal_decay_params,
                        'lr': self.optim_args['modal_lr'],
                        'weight_decay': self.optim_args['modal_decay']
                    },
                    {
                        'params': recsys_decay_params,
                        'lr': self.optim_args['rec_lr'],
                        'weight_decay': self.optim_args['rec_decay']
                    }
                ])
                optim_output = set_color(f'recsys_decay_params_len: {len(recsys_decay_params)}  modal_params_decay_len: {len(modal_decay_params)}', 'blue')
                self.logger.info(optim_output)
            else:
                optimizer = optim.AdamW([
                    {
                        'params': modal_params,
                        'lr': self.optim_args['modal_lr'],
                        'weight_decay': self.optim_args['modal_decay']
                    },
                    {
                        'params': recsys_params,
                        'lr': self.optim_args['rec_lr'],
                        'weight_decay': self.optim_args['rec_decay']
                    }
                ])
                optim_output = set_color(f'recsys_lr_params_len: {len(recsys_params)}  modal_lr_params_len: {len(modal_params)}', 'blue')
                self.logger.info(optim_output)
        elif self.config['lr_mult_prefix'] and self.config['lr_mult_rate']:
            # 匹配前缀的参数：使用 base_lr * mult_rate
            normal_params_dict = {
                "params": [],
                "lr": self.optim_args['learning_rate'],
                "weight_decay": self.optim_args['weight_decay']
            }
            high_lr_params_dict = {
                "params": [],
                "lr": self.optim_args['learning_rate'] * self.config['lr_mult_rate'],
                "weight_decay": self.optim_args['weight_decay']
            }
            self.logger.info(f'Use higher lr rate {self.config["lr_mult_rate"]} x {self.optim_args["learning_rate"]} for prefix {self.config["lr_mult_prefix"]}')

            for n, p in self.model.named_parameters():
                if any(n.startswith(x) for x in self.config['lr_mult_prefix']):
                    self.logger.info(f"high lr param: {n} {self.optim_args['learning_rate'] * self.config['lr_mult_rate']}")
                    high_lr_params_dict["params"].append(p)
                else:
                    normal_params_dict["params"].append(p)
            optimizer = optim.AdamW([normal_params_dict, high_lr_params_dict])
        elif self.config['optimizer_kwargs']:
            params = self.model.parameters()
            self.config['optimizer_kwargs']['optimizer']['params']['lr'] = self.optim_args['learning_rate']
            self.config['optimizer_kwargs']['optimizer']['params']['weight_decay'] = self.optim_args['weight_decay']
            optimizer = deepspeed.ops.adam.cpu_adam.DeepSpeedCPUAdam(params, **self.config['optimizer_kwargs']['optimizer']['params'])
        else:
            params = self.model.parameters()
            optimizer = optim.AdamW(params, lr=self.optim_args['learning_rate'], weight_decay=self.optim_args['weight_decay'])
        return optimizer

    def _build_smaller_optimizer(self):
        self.logger.info(">>> 使用 bitsandbytes 8-bit 优化器以节省显存 <<<")
        
        # 筛选需要梯度的参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # 使用 8-bit AdamW
        # 这会将优化器状态压缩 75%，从而无需 Offload 也能塞进显存
        # optimizer = bnb.optim.AdamW8bit(
        #     trainable_params,
        #     lr=self.optim_args['learning_rate'],
        #     weight_decay=self.optim_args['weight_decay']
        # )
        
        optimizer = bnb.optim.PagedAdamW8bit(
            trainable_params,
            lr=self.optim_args['learning_rate'],
            weight_decay=self.optim_args['weight_decay']
        )
        return optimizer
    
    def _train_epoch(self, train_data, epoch_idx, show_progress=False):
        self.logger.info("into _train_epoch")
        self.model.train()
        total_loss = 0
        if self.rank == 0:
            # 只有主进程显示进度条（分布式训练）
            pbar = tqdm(
                total=len(train_data),
                miniters=self.update_interval,
                desc=set_color(f"Train [{epoch_idx:>3}/{self.epochs:>3}]", 'pink'),
                file=sys.stdout
            )
        bwd_time = t.time()
        for batch_idx, data in enumerate(train_data):
            start_time = bwd_time
            self.optimizer.zero_grad()
            data = self.to_device(data)
            data_time = t.time()
            losses = self.model(data)
            fwd_time = t.time()
            #特殊损失处理（NCE损失） 噪声对比估计损失，是一种用于训练模型区分真实数据和噪声数据的损失函数
            if self.config['loss'] and self.config['loss'] == 'nce':
                model_out = losses
                losses = model_out.pop('loss')
            self._check_nan(losses)
            total_loss = total_loss + losses.item()
            self.lite.backward(losses)
            grad_norm = self.optimizer.step()
            bwd_time = t.time()
            if self.scheduler_config:
                self.lr_scheduler.step()
            if show_progress and self.rank == 0 and batch_idx % self.update_interval == 0:
                # 构建进度信息
                msg = f"loss: {losses:.4f} data: {data_time-start_time:.3f} fwd: {fwd_time-data_time:.3f} bwd: {bwd_time-fwd_time:.3f}"
                if self.scheduler_config:
                    # 添加学习率信息
                    msg = f"lr: {self.lr_scheduler.get_lr()[0]:.7f} " + msg
                if self.config['loss'] and self.config['loss'] == 'nce':
                    # 添加NCE损失的子损失信息
                    for k, v in model_out.items():
                        if k.endswith('loss'):
                            msg += f" {k}: {v:.3f}"
                if grad_norm:
                    msg = msg + f" grad_norm: {grad_norm.sum():.4f}"
                pbar.set_postfix_str(msg, refresh=False)
                pbar.update(self.update_interval)
                self.logger.info("\n" + "-"*50)
            if self.config['debug'] and batch_idx >= 10:
                break
            print(f"batch_idx:{batch_idx} waiting for input")
            self.logger.info(f"Total Reserved Mem (PyTorch Cache): {torch.cuda.memory_reserved():.2f} MB")
        return total_loss

    def _valid_epoch(self, valid_data, show_progress=False):
        #分布式同步屏障
        torch.distributed.barrier()
        valid_result = self.evaluate(valid_data, load_best_model=False, show_progress=show_progress)
        #计算验证分数
        valid_score = calculate_valid_score(valid_result, self.valid_metric)
        torch.distributed.barrier()
        return valid_score, valid_result
    #保存模型、优化器、配置等完整训练状态 记录随机数状态以保证可复现性 在分布式环境中只由主进程执行保存操作 提供详细的日志记录
    def _save_checkpoint(self, epoch, verbose=True):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id

        """
        state = {
            "model": self.model,
            "optimizer": self.optimizer,
            'config': self.config,
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_valid_score': self.best_valid_score,
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state()
        }

        self.lite.save(os.path.join(self.checkpoint_dir, self.saved_model_name), state=state)
        if self.rank == 0 and verbose:
            self.logger.info(set_color('Saving current', 'blue') + f': {self.saved_model_file}')

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    #输出训练中的损失
    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        #从配置中获取损失值显示的小数位数，如果没有配置则默认为4位
        des = self.config['loss_decimal_place'] or 4
        train_loss_output = (set_color('epoch %d training', 'green') + ' [' + set_color('time', 'blue') +
                             ': %.2fs, ') % (epoch_idx, e_time - s_time)
        #多损失情况处理
        if isinstance(losses, tuple):
            des = (set_color('train_loss%d', 'blue') + ': %.' + str(des) + 'f')
            train_loss_output += ', '.join(des % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            des = '%.' + str(des) + 'f'
            train_loss_output += set_color('train loss', 'blue') + ': ' + des % losses
        return train_loss_output + ']'

    def _add_train_loss_to_tensorboard(self, epoch_idx, losses, tag='Loss/Train'):
        if isinstance(losses, tuple):
            for idx, loss in enumerate(losses):
                self.tensorboard.add_scalar(tag + str(idx), loss, epoch_idx)
        else:
            self.tensorboard.add_scalar(tag, losses, epoch_idx)

    def _add_hparam_to_tensorboard(self, best_valid_result):
        # base hparam
        hparam_dict = {
            'learning_rate': self.config['learning_rate'],
            'weight_decay': self.config['weight_decay'],
            'train_batch_size': self.config['train_batch_size']
        }
        # unrecorded parameter
        unrecorded_parameter = {
            parameter
            for parameters in self.config.parameters.values() for parameter in parameters
        }.union({'model', 'dataset', 'config_files', 'device'})
        # other model-specific hparam
        hparam_dict.update({
            para: val
            for para, val in self.config.final_config_dict.items() if para not in unrecorded_parameter
        })
        for k in hparam_dict:
            k = k.replace('@', '_')
            if hparam_dict[k] is not None and not isinstance(hparam_dict[k], (bool, str, float, int)):
                hparam_dict[k] = str(hparam_dict[k])

        self.tensorboard.add_hparams(hparam_dict, {'hparam/best_valid_result': best_valid_result})

    #不同类型的数据输入设备的方式 现在无需区分是什么类型的数据了，这里 todevice重写
    def to_device(self, data):
        device = self.device
        if isinstance(data, tuple) or isinstance(data, list):
            tdata = ()
            for d in data:
                d = d.to(device)
                tdata += (d,)
            return tdata
        elif isinstance(data, dict):
            for k, v in data.items():
                data[k] = v.to(device)
            return data
        else:
            return data.to(device)

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        if self.scheduler_config:
            #学习率调度器设置
            warmup_rate = self.scheduler_config.get('warmup', 0.001)
            tot_steps = len(train_data) * self.epochs
            warmup_steps = tot_steps * warmup_rate
            self.lr_scheduler = self._build_scheduler(warmup_steps=warmup_steps, tot_steps=tot_steps)

        #分布式训练初始化
        world_size, local_world_size = int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_WORLD_SIZE'])
        nnodes = world_size // local_world_size
        print(f"nnodes: {nnodes}")
        precision = self.config['precision'] if self.config['precision'] else '32'
        #分布式的策略
        if self.config['strategy'] == 'deepspeed':
            self.logger.info(f"Use deepspeed strategy")
            if self.config['cpu_optimizer']:
                # 【核心修改】：针对单卡优化配置
                # stage=2 + offload_optimizer=True 是单卡性价比最高的配置（速度快，省显存）
                # 如果显存依然不够，可以尝试 stage=3 + offload_parameters=True (速度会变慢)
                strategy = DeepSpeedStrategy(
                stage=self.config.get("stage", 2), 
                precision=precision,
                offload_optimizer=True,       # 开启：将优化器状态存到 CPU 内存
                offload_parameters=False,     # 可选：如果是 stage 3，开启此项可进一步将参数存到 CPU
                offload_params_device='cpu'   # 明确指定卸载到 CPU
                )
                self.lite = L.Fabric(accelerator='gpu', devices=1, strategy=strategy, precision=precision, num_nodes=1)
            elif self.config['use_8-bit_optim']:
                # 使用 8-bit 优化器时，推荐使用 DeepSpeed Stage 2
                strategy = DeepSpeedStrategy(stage=self.config["stage"], precision=precision)
                self.lite = L.Fabric(accelerator='gpu', strategy=strategy, precision=precision, num_nodes=nnodes)
            else:
                strategy = DeepSpeedStrategy(stage=self.config["stage"], precision=precision)
                self.lite = L.Fabric(accelerator='gpu', strategy=strategy, precision=precision, num_nodes=nnodes)

        else:
            self.logger.info(f"Use DDP strategy")
            strategy = DDPStrategy(find_unused_parameters=True)
            self.lite = L.Fabric(accelerator='gpu', strategy=strategy, precision=precision, num_nodes=nnodes)
        self.lite.launch()

        # 如果是为了显存使用 cpu 计算优化器的话 传入的优化器必须是 DeepSpeedCPUAdam
        self.model, self.optimizer = self.lite.setup(self.model, self.optimizer)
        
        torch.cuda.empty_cache() # 强制释放缓存
        print("Cache emptied to release fragmentation.")
        
        self.logger.info(f"finish load model")
        base_mem = torch.cuda.memory_allocated() / 1024**2
        print(f"fit Loop Start Mem: {base_mem:.2f} MB")   #fit Loop Start Mem: 12611.09 MB
        # 添加这行代码来验证
        reserved_mem = torch.cuda.memory_reserved() / 1024**2
        print(f"Total Reserved Mem (PyTorch Cache): {reserved_mem:.2f} MB")
        # input()
        #自动恢复机制
        if self.config['auto_resume']:
            raise NotImplementedError

        valid_step = 0

        #主训练循环
        for epoch_idx in range(self.start_epoch, self.epochs):
            self.logger.info(f"epoch_idx:{epoch_idx}")
            self.logger.info(f"Total Reserved Mem (PyTorch Cache): {torch.cuda.memory_reserved():.2f} MB")
            # train
            if self.config['need_training'] == None or self.config['need_training']:
                # 每个 epoch 中数据的随机洗牌（Shuffle）方式不同
                train_data.sampler.set_epoch(epoch_idx)

                training_start_time = time()
                train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
                self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
                training_end_time = time()
                train_loss_output = \
                    self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
                
                if verbose:
                    self.logger.info(train_loss_output)
                if self.rank == 0:
                    self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
                self.wandblogger.log_metrics({'epoch': epoch_idx, 'train_loss': train_loss, 'train_step': epoch_idx}, head='train')
            #验证和早停机制
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue
            #控制验证频率，避免每个epoch都验证
            #示例：如果eval_step = 5，则在epoch 4, 9, 14...时进行验证
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)
                #早停判断
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger
                )

                valid_end_time = time()
                valid_score_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                      + ": %.2fs, " + set_color("valid_score", 'blue') + ": %f]") % \
                    (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = set_color('valid result', 'blue') + ': \n' + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                if self.rank == 0:
                    self.tensorboard.add_scalar('Vaild_score', valid_score, epoch_idx)
                    for name, value in valid_result.items():
                        self.tensorboard.add_scalar(name.replace('@', '_'), value, epoch_idx)
                self.wandblogger.log_metrics({**valid_result, 'valid_step': valid_step}, head='valid')

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                        (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step += 1

        return self.best_valid_score, self.best_valid_result


    # user:用户标识
    # time_seq::时间序列(可能用于时序建模)
    # history_index:用户历史交互过的物品索引（用于排除已交互物品）
    # positive_u / positive_i:正样本用户和物品（用于评估）
    
    @torch.no_grad()
    def _full_sort_batch_eval(self, batched_data):
        print(f"into _full_sort_batch_eval")
        user, time_seq, history_index, positive_u, positive_i = batched_data
        interaction = self.to_device(user)
        time_seq = self.to_device(time_seq)

        # 【修改点 1】：处理 item_feature 的设备问题
        # 此时 self.item_feature 存储在 CPU 上，但模型在 GPU 上。
        # 我们需要在推理前将其移动到 GPU。
        # 注意：如果显存非常紧张，移动整个大张量可能会再次 OOM，但在拼接阶段已经省下了大量显存，通常这里能放下。
        # if isinstance(self.item_feature, tuple):
        #     # 如果是元组（如图像+文本特征），分别移动到 GPU
        #     batch_item_feature = tuple(x.to(self.device) for x in self.item_feature)
        # else:
        #     # 如果是单个张量，直接移动到 GPU
        #     batch_item_feature = self.item_feature.to(self.device)

        # if self.config['model'] == 'HLLM':
        #     # HLLM 且处于第 3 阶段（部署阶段），使用 self.model.module.predict 进行推理
        #     if self.config['stage'] == 3:
        #         scores = self.model.module.predict(interaction, time_seq, batch_item_feature)
        #     else:
        #         scores = self.model((interaction, time_seq, batch_item_feature), mode='predict')
        # else:
        #     scores = self.model.module.predict(interaction, time_seq, batch_item_feature)

        if self.config['model'] == 'HLLM':
            #HLLM 且处于第 3 阶段（部署阶段），使用 self.model.module.predict 进行推理
            if self.config['stage'] == 3:
                scores = self.model.module.predict(interaction, time_seq, self.item_feature)
            else:
                scores = self.model((interaction, time_seq, self.item_feature), mode='predict')
        else:
            scores = self.model.module.predict(interaction, time_seq, self.item_feature)
        scores = scores.view(-1, self.tot_item_num)
        scores[:, 0] = -np.inf
        if history_index is not None:
            #将历史交互过的物品的分数也设为负无穷（避免推荐已经交互过的物品）
            scores[history_index] = -np.inf
        return scores, positive_u, positive_i

    # 实现的是物品特征预计算(compute_item_feature)功能
    # 利用HLLM中的Item LLM将物品文本描述转换为固定维度的嵌入向量

    @torch.no_grad()
    def compute_item_feature(self, config, data):
        if self.use_text:
            item_data = BatchTextDataset(config, data)
            item_batch_size = config['MAX_ITEM_LIST_LENGTH'] * config['train_batch_size']
            item_loader = DataLoader(item_data, batch_size=item_batch_size, num_workers=14, shuffle=False, pin_memory=True, collate_fn=customize_rmpad_collate)
            self.logger.info(f"Inference item_data with {item_batch_size = } {len(item_loader) = }")
            self.item_feature = []
            with torch.no_grad():
                #enumerate(item_loader) 为每个批次添加索引 返回 (index, batch_data) 元组
                for idx, items in tqdm(enumerate(item_loader), total=len(item_loader)):
                    items = self.to_device(items)
                    items = self.model(items, mode='compute_item')

                    # 【修改点 2】：计算完成后立即移回 CPU
                    # 目的：避免 GPU 显存中积累大量计算图或张量，释放显存给后续批次
                    # if isinstance(items, tuple):
                    #     items = tuple(x.cpu() for x in items)
                    # else:
                    #     items = items.cpu()

                    self.item_feature.append(items)
                # 处理单输出或多输出情况
                # 物品可能有多种特征：图像特征 + 文本特征  这里是尝试将特征划分开了
                if isinstance(items, tuple):
                    self.item_feature = torch.cat([x[0] for x in self.item_feature]), torch.cat([x[1] for x in self.item_feature])
                else:
                    self.item_feature = torch.cat(self.item_feature)
                if self.config['stage'] == 3:
                    self.item_feature = self.item_feature.bfloat16()


                # 【修改点 3】：拼接操作在 CPU 上进行
                # 因为 list 中的元素已经是 CPU 张量了，torch.cat 会在 CPU 内存中分配空间，不占用显存
                # if isinstance(self.item_feature[0], tuple): # 检查第一个元素是否为 tuple
                #     # 假设结构是 [(feat1_batch1, feat2_batch1), (feat1_batch2, feat2_batch2), ...]
                #     # 我们需要将其转变为 (cat(feat1_all), cat(feat2_all))
                #     feat1 = torch.cat([x[0] for x in self.item_feature])
                #     feat2 = torch.cat([x[1] for x in self.item_feature])
                #     self.item_feature = (feat1, feat2)
                # else:
                #     self.item_feature = torch.cat(self.item_feature)
                # # 类型转换也在 CPU 上完成
                # if self.config['stage'] == 3:
                #     if isinstance(self.item_feature, tuple):
                #         self.item_feature = tuple(x.bfloat16() for x in self.item_feature)
                #     else:
                #         self.item_feature = self.item_feature.bfloat16()
        else:
            with torch.no_grad():
                self.item_feature = self.model.module.compute_item_all()

    #将多个GPU/进程上的张量收集起来，计算全局平均值。
    def distributed_concat(self, tensor, num_total_examples):
        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)
        concat = torch.cat(output_tensors, dim=0)
        return concat.sum() / num_total_examples

    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False, init_model=False):
        if not eval_data:
            return
        if init_model:
            world_size, local_world_size = int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_WORLD_SIZE'])
            nnodes = world_size // local_world_size
            if self.config['strategy'] == 'deepspeed':
                self.logger.info(f"Use deepspeed strategy")
                # input()
                # 精度设置  precision = bf16-mixed
                precision = self.config['precision'] if self.config['precision'] else '32'
                # stage = 2     DeepSpeed 的优化阶段
                strategy = DeepSpeedStrategy(stage=self.config['stage'], precision=precision)
                #  Fabric 初始化
                self.lite = L.Fabric(accelerator='gpu', strategy=strategy, precision=precision, num_nodes=nnodes)
                # 启动分布式环境
                self.lite.launch()
                self.model, self.optimizer = self.lite.setup(self.model, self.optimizer)
                now_mem = torch.cuda.memory_allocated() / 1024**2
                print(f"finish load mem: {now_mem:.2f} MB")
                print("finished init model")
            else:
                self.logger.info(f"Use DDP strategy")
                precision = self.config['precision'] if self.config['precision'] else '32'
                strategy = DDPStrategy(find_unused_parameters=True)
                self.lite = L.Fabric(accelerator='gpu', strategy=strategy, precision=precision, num_nodes=nnodes)
                self.lite.launch()
                self.model = self.lite.setup(self.model)

        #加载训练好的最佳模型参数
        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            state = {"model": self.model}
            self.lite.load(checkpoint_file, state)
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            self.logger.info(message_output)

        print("start evalute")
        with torch.no_grad():   
            self.model.eval()
            print("into eval no grad")
            # 评估函数的方法，这里采用的是全排序的   
            eval_func = self._full_sort_batch_eval

            self.tot_item_num = eval_data.dataset.dataload.item_num
            # 卡在这一步的执行上面了   这里把物品的 embending存入显存了
            self.compute_item_feature(self.config, eval_data.dataset.dataload)
            print("finish compute_item_feature")
            iter_data = (
                tqdm(
                    eval_data,
                    total=len(eval_data),
                    ncols=150,
                    desc=set_color(f"Evaluate   ", 'pink'),
                    file=sys.stdout
                ) if show_progress and self.rank == 0 else eval_data
            )

            # 批量推理
            fwd_time = t.time()
            # # 【监控】循环开始前的显存
            base_mem = torch.cuda.memory_allocated() / 1024**2
            print(f"Loop Start Mem: {base_mem:.2f} MB")
            for batch_idx, batched_data in enumerate(iter_data):
                start_time = fwd_time
                data_time = t.time()
                scores, positive_u, positive_i = eval_func(batched_data)
                fwd_time = t.time()
                if show_progress and self.rank == 0:
                    iter_data.set_postfix_str(f"data: {data_time-start_time:.3f} fwd: {fwd_time-data_time:.3f}", refresh=False)
                self.eval_collector.eval_batch_collect(scores, positive_u, positive_i)

            num_total_examples = len(eval_data.sampler.dataset)
            struct = self.eval_collector.get_data_struct()
            result = self.evaluator.evaluate(struct)

            metric_decimal_place = 5 if self.config['metric_decimal_place'] == None else self.config['metric_decimal_place']
            for k, v in result.items():
                result_cpu = self.distributed_concat(torch.tensor([v]).to(self.device), num_total_examples).cpu()
                result[k] = round(result_cpu.item(), metric_decimal_place)
            self.wandblogger.log_eval_metrics(result, head='eval')

            return result

            

'''
import os
import sys
from logging import getLogger
from time import time
import time as t
import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from tqdm import tqdm
import bitsandbytes as bnb  # <--- 必须确保安装了 bitsandbytes

from ..data.dataset import BatchTextDataset
from ..data.dataset.collate_fn import customize_rmpad_collate
from torch.utils.data import DataLoader
from ..evaluator import Evaluator, Collector
from ..utils import ensure_dir, get_local_time, early_stopping, calculate_valid_score, dict2str, \
    get_tensorboard, set_color, get_gpu_usage, WandbLogger
from ..utils.lr_scheduler import *

import lightning as L
from lightning.fabric.strategies import DeepSpeedStrategy, DDPStrategy

class Trainer(object):
    def __init__(self, config, model):
        super(Trainer, self).__init__()
        self.config = config
        self.model = model
        self.logger = getLogger()
        self.wandblogger = WandbLogger(config)
        self.optim_args = config['optim_args']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config.get('clip_grad_norm', 1.0)
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.gpu_available = torch.cuda.is_available() and config['use_gpu']
        self.device = config['device']
        self.rank = torch.distributed.get_rank()
        if self.rank == 0:
            self.tensorboard = get_tensorboard(self.logger)

        self.checkpoint_dir = config['checkpoint_dir']
        if self.rank == 0:
            ensure_dir(self.checkpoint_dir)

        self.saved_model_name = '{}-{}.pth'.format(self.config['model'], 0)
        self.saved_model_file = os.path.join(self.checkpoint_dir, self.saved_model_name)
        self.use_text = config['use_text']
        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.train_loss_dict = dict()
        
        # === [核心修复 1] 强制使用 8-bit 优化器构建函数 ===
        self.optimizer = self._build_optimizer()
        
        self.update_interval = config['update_interval'] if config['update_interval'] else 20
        self.scheduler_config = config['scheduler_args']

        # === [核心修复 2] freeze_prefix 鲁棒性处理 (解决 extend 报错) ===
        if config['freeze_prefix'] or config['freeze_ad']:
            freeze_prefix = config['freeze_prefix'] if config['freeze_prefix'] else []
            
            # 强制转为列表，防止传入字符串导致报错
            if isinstance(freeze_prefix, str):
                # 尝试解析潜在的列表字符串
                if freeze_prefix.startswith('[') and freeze_prefix.endswith(']'):
                    try:
                        import json
                        # 替换单引号为双引号以符合 JSON 标准
                        freeze_prefix = json.loads(freeze_prefix.replace("'", '"'))
                    except:
                        freeze_prefix = [freeze_prefix]
                else:
                    freeze_prefix = [freeze_prefix]
            
            if config['freeze_ad']:
                freeze_prefix.extend(['item_llm', 'item_emb_tokens'])
            if not config['ft_item']:
                freeze_prefix.extend(['item_embedding'])
            self._freeze_params(freeze_prefix)

        for n, p in self.model.named_parameters():
            self.logger.info(f"{n} {p.size()} {p.requires_grad}")

        self.eval_collector = Collector(config)
        self.evaluator = Evaluator(config)
        self.item_feature = None
        self.tot_item_num = None

    def _freeze_params(self, freeze_prefix):
        for name, param in self.model.named_parameters():
            for prefix in freeze_prefix:
                if name.startswith(prefix):
                    self.logger.info(f"freeze_params: {name}")
                    param.requires_grad = False

    def _build_scheduler(self, warmup_steps=None, tot_steps=None):
        if self.scheduler_config['type'] == 'cosine':
            return get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, tot_steps)
        elif self.scheduler_config['type'] == 'liner':
            return get_linear_schedule_with_warmup(self.optimizer, warmup_steps, tot_steps)
        else:
            return get_constant_schedule(self.optimizer)

    def _build_optimizer(self):
        # === [核心修复 3] 移除旧逻辑，强制启用 8-bit AdamW ===
        self.logger.info(">>> [System] Forcing use of 8-bit AdamW (bitsandbytes) to save VRAM <<<")
        
        # 筛选出需要梯度的参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # 使用 8-bit 优化器
        # optimizer = bnb.optim.AdamW8bit(
        #     trainable_params,
        #     lr=self.optim_args['learning_rate'],
        #     weight_decay=self.optim_args['weight_decay']
        # )
        # 使用 Paged 优化器
        optimizer = bnb.optim.PagedAdamW8bit(
            trainable_params,
            lr=self.optim_args['learning_rate'],
            weight_decay=self.optim_args['weight_decay']
        )
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, show_progress=False):
        self.logger.info("into _train_epoch")
        self.model.train()
        total_loss = 0
        if self.rank == 0:
            pbar = tqdm(total=len(train_data), miniters=self.update_interval, desc=set_color(f"Train [{epoch_idx:>3}/{self.epochs:>3}]", 'pink'), file=sys.stdout)
        
        bwd_time = t.time()
        for batch_idx, data in enumerate(train_data):
            # [监控] 打印显存状态，确认优化器 Step 时的行为
            if batch_idx == 0:
                 mem_before = torch.cuda.memory_allocated() / 1024**2
                 print(f">>> [Batch 0] Mem before step: {mem_before:.2f} MB")

            start_time = bwd_time
            self.optimizer.zero_grad()
            data = self.to_device(data)
            data_time = t.time()
            
            losses = self.model(data)
            fwd_time = t.time()
            
            if self.config['loss'] and self.config['loss'] == 'nce':
                model_out = losses
                losses = model_out.pop('loss')
            
            self._check_nan(losses)
            total_loss = total_loss + losses.item()
            
            self.lite.backward(losses)
            
            grad_norm = None
            if self.clip_grad_norm:
                grad_norm = self.lite.clip_gradients(self.model, self.optimizer, max_norm=self.clip_grad_norm)
            
            # 这里是之前报错的地方，更换优化器后显存占用应大幅降低
            self.optimizer.step()
            
            # [监控] Step 之后
            if batch_idx == 0:
                 mem_after = torch.cuda.memory_allocated() / 1024**2
                 print(f">>> [Batch 0] Mem after step: {mem_after:.2f} MB (Delta: {mem_after - mem_before:.2f} MB)")

            bwd_time = t.time()
            if self.scheduler_config:
                self.lr_scheduler.step()
                
            if show_progress and self.rank == 0 and batch_idx % self.update_interval == 0:
                msg = f"loss: {losses:.4f} data: {data_time-start_time:.3f} fwd: {fwd_time-data_time:.3f} bwd: {bwd_time-fwd_time:.3f}"
                if self.scheduler_config:
                    msg = f"lr: {self.lr_scheduler.get_lr()[0]:.7f} " + msg
                pbar.set_postfix_str(msg, refresh=False)
                pbar.update(self.update_interval)

            # Debug 模式
            if self.config['debug'] and batch_idx >= 10:
                break
        return total_loss

    def _valid_epoch(self, valid_data, show_progress=False):
        torch.distributed.barrier()
        valid_result = self.evaluate(valid_data, load_best_model=False, show_progress=show_progress)
        valid_score = calculate_valid_score(valid_result, self.valid_metric)
        torch.distributed.barrier()
        return valid_score, valid_result

    def _save_checkpoint(self, epoch, verbose=True):
        state = {
            "model": self.model,
            "optimizer": self.optimizer,
            'config': self.config,
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_valid_score': self.best_valid_score,
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state()
        }
        self.lite.save(os.path.join(self.checkpoint_dir, self.saved_model_name), state=state)
        if self.rank == 0 and verbose:
            self.logger.info(set_color('Saving current', 'blue') + f': {self.saved_model_file}')

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        des = self.config['loss_decimal_place'] or 4
        train_loss_output = (set_color('epoch %d training', 'green') + ' [' + set_color('time', 'blue') +
                             ': %.2fs, ') % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            des = (set_color('train_loss%d', 'blue') + ': %.' + str(des) + 'f')
            train_loss_output += ', '.join(des % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            des = '%.' + str(des) + 'f'
            train_loss_output += set_color('train loss', 'blue') + ': ' + des % losses
        return train_loss_output + ']'

    def _add_train_loss_to_tensorboard(self, epoch_idx, losses, tag='Loss/Train'):
        if isinstance(losses, tuple):
            for idx, loss in enumerate(losses):
                self.tensorboard.add_scalar(tag + str(idx), loss, epoch_idx)
        else:
            self.tensorboard.add_scalar(tag, losses, epoch_idx)

    def to_device(self, data):
        device = self.device
        if isinstance(data, dict):
            for k, v in data.items():
                data[k] = v.to(device)
            return data
        return data.to(device)

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        if self.scheduler_config:
            warmup_rate = self.scheduler_config.get('warmup', 0.001)
            tot_steps = len(train_data) * self.epochs
            warmup_steps = tot_steps * warmup_rate
            self.lr_scheduler = self._build_scheduler(warmup_steps=warmup_steps, tot_steps=tot_steps)

        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
        nnodes = world_size // local_world_size
        precision = self.config['precision'] if self.config['precision'] else '32'

        # === [核心修复 4] 强制使用 DDP 策略，因为 bitsandbytes 在 DeepSpeed 下需要特殊配置，简单起见用 DDP ===
        self.logger.info(">>> Strategy: DDP with 8-bit Optimizer <<<")
        strategy = DDPStrategy(find_unused_parameters=True)
        self.lite = L.Fabric(accelerator='gpu', strategy=strategy, precision=precision, num_nodes=nnodes)
        
        self.lite.launch()

        self.model, self.optimizer = self.lite.setup(self.model, self.optimizer)
        
        torch.cuda.empty_cache()
        self.logger.info(f"finish load model")
        
        # 验证优化器类型
        print(f"Optimizer Class: {type(self.optimizer)}")
        input()
        for epoch_idx in range(self.start_epoch, self.epochs):
            if self.config['need_training'] == None or self.config['need_training']:
                train_data.sampler.set_epoch(epoch_idx)
                training_start_time = time()
                train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
                self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
                training_end_time = time()
                train_loss_output = self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
                
                if verbose:
                    self.logger.info(train_loss_output)
                if self.rank == 0:
                    self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
                self.wandblogger.log_metrics({'epoch': epoch_idx, 'train_loss': train_loss, 'train_step': epoch_idx}, head='train')

            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue

            if (epoch_idx + 1) % self.eval_step == 0:
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score, self.best_valid_score, self.cur_step, max_step=self.stopping_step, bigger=self.valid_metric_bigger
                )
                valid_result_output = set_color('valid result', 'blue') + ': \n' + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_result_output)
                self.wandblogger.log_metrics({**valid_result}, head='valid')

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result
                
                if stop_flag:
                    break

        return self.best_valid_score, self.best_valid_result

    #  (后续 evaluate 等代码可以保持原样，或者你之前发给我的那部分) 
    @torch.no_grad()
    def _full_sort_batch_eval(self, batched_data):
        user, time_seq, history_index, positive_u, positive_i = batched_data
        interaction = self.to_device(user)
        time_seq = self.to_device(time_seq)
        if isinstance(self.item_feature, tuple):
            batch_item_feature = tuple(x.to(self.device) for x in self.item_feature)
        else:
            batch_item_feature = self.item_feature.to(self.device)

        scores = self.model.module.predict(interaction, time_seq, batch_item_feature)
        scores = scores.view(-1, self.tot_item_num)
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf
        return scores, positive_u, positive_i

    @torch.no_grad()
    def compute_item_feature(self, config, data):
        if self.use_text:
            item_data = BatchTextDataset(config, data)
            item_batch_size = config['MAX_ITEM_LIST_LENGTH'] * config['train_batch_size']
            item_loader = DataLoader(item_data, batch_size=item_batch_size, num_workers=2, shuffle=False, pin_memory=True, collate_fn=customize_rmpad_collate)
            self.logger.info(f"Inference item_data with {item_batch_size = } {len(item_loader) = }")
            self.item_feature = []
            with torch.no_grad():
                for idx, items in tqdm(enumerate(item_loader), total=len(item_loader)):
                    items = self.to_device(items)
                    items = self.model(items, mode='compute_item')
                    if isinstance(items, tuple):
                        items = tuple(x.cpu() for x in items)
                    else:
                        items = items.cpu()
                    self.item_feature.append(items)
                
                if isinstance(self.item_feature[0], tuple):
                    feat1 = torch.cat([x[0] for x in self.item_feature])
                    feat2 = torch.cat([x[1] for x in self.item_feature])
                    self.item_feature = (feat1, feat2)
                else:
                    self.item_feature = torch.cat(self.item_feature)
        else:
            with torch.no_grad():
                self.item_feature = self.model.module.compute_item_all()

    def distributed_concat(self, tensor, num_total_examples):
        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)
        concat = torch.cat(output_tensors, dim=0)
        return concat.sum() / num_total_examples

    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False, init_model=False):
        if not eval_data:
            return
        if init_model:
            world_size = int(os.environ.get('WORLD_SIZE', 1))
            local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
            nnodes = world_size // local_world_size
            precision = self.config['precision'] if self.config['precision'] else '32'
            strategy = DDPStrategy(find_unused_parameters=True)
            self.lite = L.Fabric(accelerator='gpu', strategy=strategy, precision=precision, num_nodes=nnodes)
            self.lite.launch()
            self.model = self.lite.setup(self.model)

        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            state = {"model": self.model}
            self.lite.load(checkpoint_file, state)
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            self.logger.info(message_output)

        with torch.no_grad():   
            self.model.eval()
            eval_func = self._full_sort_batch_eval
            self.tot_item_num = eval_data.dataset.dataload.item_num
            self.compute_item_feature(self.config, eval_data.dataset.dataload)
            iter_data = (
                tqdm(eval_data, total=len(eval_data), ncols=150, desc=set_color(f"Evaluate   ", 'pink'), file=sys.stdout)
                if show_progress and self.rank == 0 else eval_data
            )
            for batch_idx, batched_data in enumerate(iter_data):
                scores, positive_u, positive_i = eval_func(batched_data)
                self.eval_collector.eval_batch_collect(scores, positive_u, positive_i)

            num_total_examples = len(eval_data.sampler.dataset)
            struct = self.eval_collector.get_data_struct()
            result = self.evaluator.evaluate(struct)
            metric_decimal_place = 5 if self.config['metric_decimal_place'] == None else self.config['metric_decimal_place']
            for k, v in result.items():
                result_cpu = self.distributed_concat(torch.tensor([v]).to(self.device), num_total_examples).cpu()
                result[k] = round(result_cpu.item(), metric_decimal_place)
            self.wandblogger.log_eval_metrics(result, head='eval')
            return result
        
'''