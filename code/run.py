# Copyright (c) 2024 westlake-repl
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
# SPDX-License-Identifier: MIT
# This file has been modified by Junyi Chen.
#
# Original file was released under MIT, with the full license text
# available at https://choosealicense.com/licenses/mit/.
#
# This modified file is released under the same license.

from cProfile import run
from logging import getLogger
import torch
import json
from REC.data import *
from REC.config import Config
from REC.utils import init_logger, get_model, init_seed, set_color
from REC.trainer import Trainer
import torch.distributed as dist

import os
import numpy as np
import argparse
import torch.distributed as dist
import torch
from REC.utils import print_gpu_memory

def convert_str(s):
    try:
        if s.lower() == 'none':
            return None
        if s.lower() == 'true':
            return True
        if s.lower() == 'false':
            return False
        float_val = float(s)
        if float_val.is_integer():
            return int(float_val)
        return float_val
    except ValueError:
        print(f"Unable to convert the string '{s}' to None / Bool / Float / Int, retaining the original string.")
        return s


def run_loop(local_rank, config_file=None, saved=True, extra_args=[]):

    # configurations initialization
    # print(f"config_file :{config_file}")
    # input()
    config = Config(config_file_list=config_file)
    # print(f"config:{config}")
    
    device = torch.device("cuda", local_rank)
    config['device'] = device
    if len(extra_args):
        for i in range(0, len(extra_args), 2):
            key = extra_args[i][2:]
            value = extra_args[i + 1]
            try:
                if '[' in value or '{' in value:
                    value = json.loads(value)
                    if isinstance(value, dict):
                        for k, v in value.items():
                            value[k] = convert_str(v)
                    else:
                        value = [convert_str(x) for x in value]
                else:
                    value = convert_str(value)
                if '.' in key:
                    k1, k2 = key.split('.')
                    config[k1][k2] = value
                else:
                    config[key] = value
            except:
                raise ValueError(f"{key} {value} invalid")

    init_seed(config['seed'], config['reproducibility'])
    # input()
    # logger initialization
    init_logger(config)
    logger = getLogger()
    if 'text_path' in config:
        config['text_path'] = os.path.join(config['text_path'], config['dataset'] + '.csv')
        logger.info(f"Update text_path to {config['text_path']}")

    #print(f" config information:{config}")
    '''
    config information:
    General Hyper Parameters:
    seed = 2020
    state = INFO
    use_text = True
    reproducibility = True
    checkpoint_dir = /root/autodl-pvt/HLLM/checkpoints
    show_progress = True
    log_wandb = False
    data_path = ../dataset/
    strategy = deepspeed
    precision = bf16-mixed
    model = HLLM

    Training Hyper Parameters:
    epochs = 5
    train_batch_size = 2
    optim_args = {'learning_rate': 0.0001, 'weight_decay': 0.01}
    eval_step = 1
    stopping_step = 5

    Evaluation Hyper Parameters:
    eval_batch_size = 256
    topk = [5, 10, 50, 200]
    metrics = ['Recall', 'NDCG']
    valid_metric = NDCG@200
    metric_decimal_place = 7
    eval_type = EvaluatorType.RANKING
    valid_metric_bigger = True

    Dataset Hyper Parameters:
    MAX_ITEM_LIST_LENGTH = 10
    MAX_TEXT_LENGTH = 256
    text_keys = ['title', 'description']
    item_prompt = Compress the following sentence into embedding: 

    Other Hyper Parameters: 
    wandb_project = REC
    text_path = /root/autodl-pvt/HLLM/information/amazon_books.csv
    item_emb_token_n = 1
    loss = nce
    scheduler_args = {'type': 'cosine', 'warmup': 0.1}
    stage = 2
    item_pretrain_dir = TinyLlama/TinyLlama-1.1B-Chat-v1.0
    item_llm_init = True
    user_pretrain_dir = TinyLlama/TinyLlama-1.1B-Chat-v1.0
    user_llm_init = True
    use_ft_flash_attn = True
    MODEL_INPUT_TYPE = InputType.SEQ
    device = cuda:0
    val_only = True
    '''

    # get model and data
    dataload = load_data(config)
    train_loader, valid_loader, test_loader = bulid_dataloader(config, dataload)
    print(f"{len(train_loader) = }")

    model = get_model(config['model'])(config, dataload)
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    world_size = torch.distributed.get_world_size()
    trainer = Trainer(config, model)

    logger.info(set_color('\nWorld_Size', 'pink') + f' = {world_size} \n')
    logger.info(config)
    logger.info(dataload)
    logger.info('model structure:')
    logger.info(model)
    print("rec/run.py")
    print("waiting for input")
    print_gpu_memory()
    # input()   # 到这里显存使用都不大

    # 典型的机器学习模型训练和评估流程
    if config['val_only']:
        # 只进行模型评估，不训练
        # 1. 加载预训练模型
        ckpt_path = os.path.join(config['checkpoint_dir'], 'pytorch_model.bin')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        logger.info(f'Eval only model load from {ckpt_path}')
        # 2. 加载模型权重
        msg = trainer.model.load_state_dict(ckpt, False)
        logger.info(f'{msg.unexpected_keys = }')
        logger.info(f'{msg.missing_keys = }')
        # 3. 在测试集上评估
        test_result = trainer.evaluate(
            test_loader, 
            load_best_model=False,    # 不加载最佳模型（因为已经手动加载）
            show_progress=config['show_progress'],
            init_model=True           # 初始化模型状态
        )
        logger.info(set_color('test result', 'yellow') + f': {test_result}')
    else:
        # 完整训练模式 
        logger.info('start training')
        # training process
        best_valid_score, best_valid_result = trainer.fit(
            train_loader, valid_loader, saved=saved, show_progress=config['show_progress']
        )
        logger.info(f'Trianing Ended' + set_color('best valid ', 'yellow') + f': {best_valid_result}')

        # model evaluation
        test_result = trainer.evaluate(test_loader, load_best_model=saved, show_progress=config['show_progress'])

        logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
        logger.info(set_color('test result', 'yellow') + f': {test_result}')

        return {
            'best_valid_score': best_valid_score,
            'valid_score_bigger': config['valid_metric_bigger'],
            'best_valid_result': best_valid_result,
            'test_result': test_result
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", nargs='+', type=str)
    args, extra_args = parser.parse_known_args()
    local_rank = int(os.environ['LOCAL_RANK'])
    config_file = args.config_file

    '''
    1. PARSED ARGUMENTS:
   - config_file: ['overall/LLM_deepspeed.yaml', 'HLLM/HLLM.yaml']
   - extra_args: ['--loss', 'nce', '--epochs', '5', '--dataset', 'amazon_books', '--train_batch_size', '2', '--MAX_TEXT_LENGTH', '256', '--MAX_ITEM_LIST_LENGTH', '10', '--checkpoint_dir', '/root/autodl-pvt/HLLM/checkpoints', '--optim_args.learning_rate', '1e-4', '--item_pretrain_dir', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', '--user_pretrain_dir', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', '--text_path', '/root/autodl-pvt/HLLM/information', '--text_keys', '["title","description"]', '--val_only', 'True']
   - local_rank: 0
    '''

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    run_loop(local_rank=local_rank, config_file=config_file, extra_args=extra_args)
