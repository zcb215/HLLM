# Copyright (c) 2024 westlake-repl
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
# SPDX-License-Identifier: MIT
# This file has been modified by Junyi Chen.
#
# Original file was released under MIT, with the full license text
# available at https://choosealicense.com/licenses/mit/.
#
# This modified file is released under the same license.

from asyncio.log import logger
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import random
import datetime
import pytz
import math
import torch.distributed as dist
from datasets import load_dataset

# 数据形式为 [[user_seq], [neg_item_seq]] , [mask]


class SEQTrainDataset(Dataset):
    def __init__(self, config, dataload):
        self.dataload = dataload
        self.config = config

        self.item_num = dataload.item_num
        # 记住这里的数据集已经从字典转为列表 了 纯物品数据
        self.train_seq = dataload.train_feat['item_seq']

        self.length = len(self.train_seq)
        # 这里已经是  S + 1  所以后面的reshape 其实时实际上面没变化
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']+1
        self.device = config['device']
        self.random_sample = True if config['loss'] and config['loss'] == 'nce' else False
        self.num_negatives = config['num_negatives']
        if self.num_negatives:
            # 这就是全局 Batch Size（Global Batch Size），即一次训练迭代中所有 GPU 处理的样本总和  get_world_size 是 GPU 的数量
            self.num_negatives = math.ceil(self.num_negatives / dist.get_world_size() / config['train_batch_size'])
        logger.info(f"Use random sample {self.random_sample} for mask id")

    def __len__(self):
        return self.length

    def _neg_sample(self, item_set):
        # 疯狂的循环找到 不在 已选队列里面的
        item = random.randint(1, self.item_num - 1)
        while item in item_set:
            item = random.randint(1, self.item_num - 1)
        return item

    def _padding_sequence(self, sequence, max_length, random_sample=False):
        # 对序列进行填充
        pad_len = max_length - len(sequence)
        if random_sample:
            pad_seq = [self._neg_sample(sequence) for _ in range(pad_len)]
            sequence = pad_seq + sequence
        else:
            sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:]
        return torch.tensor(sequence, dtype=torch.long)

    def reconstruct_train_data(self, item_seq):
        masked_index = []
        neg_item = []
        item_seq_len = len(item_seq)
        # 不遍历到 item_seq_len  是因为训练的时候需要有正样本 ，数据只有这么多
        for i in range(item_seq_len - 1):
            neg_item.append(self._neg_sample(item_seq))
            masked_index.append(1)

        item_seq = self._padding_sequence(list(item_seq), self.max_seq_length, random_sample=self.random_sample)
        if self.num_negatives:
            neg_item = []
            for _ in range(self.num_negatives):
                neg_item.append(self._neg_sample(item_seq))
        else:
            neg_item = self._padding_sequence(neg_item, self.max_seq_length, random_sample=self.random_sample)
        masked_index = self._padding_sequence(masked_index, self.max_seq_length-1)
        return torch.as_tensor(item_seq, dtype=torch.int64), torch.as_tensor(neg_item, dtype=torch.int64), torch.as_tensor(masked_index, dtype=torch.int64)

    def __getitem__(self, index):
        # 最长长度为maxlen+1, 及若max_len是5
        # 则存在    1,2,3,4,5,6序列,
        # pos       2,3,4,5,6
        # neg       0,8,9,7,9,8
        # mask_index 1,1,1,1,1
        item_seq = self.train_seq[index]
        item_seq, neg_item, masked_index = self.reconstruct_train_data(item_seq)

        return item_seq, neg_item, masked_index


# class TextSEQTrainDataset(Dataset):
#     def __init__(self, config, dataload):
#         self.dataload = dataload
#         self.config = config

#         self.item_num = dataload.item_num
#         self.train_seq = dataload.train_feat['item_seq']
#         self.length = len(self.train_seq)
#         self.train_time_seq = dataload.train_feat['time_seq']
#         self.id2token = dataload.id2token['item_id']

#         self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']+1
#         self.max_text_length = config['MAX_TEXT_LENGTH']
#         self.device = config['device']

#         self.text_path = config['text_path']
#         self.text_keys = config['text_keys']
#         self.tokenizer = AutoTokenizer.from_pretrained(config['item_pretrain_dir'], trust_remote_code=True)
#         # self.pad_id = self.tokenizer.pad_token_id
#         # assert self.pad_id is not None, f"pad_token_id can't be {self.pad_id}"
#         self.item_prompt = config['item_prompt']
#         self.item_emb_token_n = config['item_emb_token_n']
#         self.num_negatives = config['num_negatives']
#         self.random_sample = True if config['loss'] and config['loss'] == 'nce' else False
#         if self.num_negatives:
#             self.num_negatives = math.ceil(self.num_negatives / dist.get_world_size() / config['train_batch_size'])  # for llm only
#         logger.info(f"Use random sample {self.random_sample} for mask id")
#         logger.info(f"Text path: {self.text_path}")
#         logger.info(f"Text keys: {self.text_keys}")
#         logger.info(f"Item prompt: {self.item_prompt}")
#         self.load_content()

#     def __len__(self):
#         return self.length

#     def load_content(self):
#         self.env = pd.read_csv(self.text_path, delimiter=',', dtype={'item_id': str})
#         self.env = self.env[self.text_keys + ['item_id']]

#         if self.config['memory_optimize']:
#             self.env = self.env.set_index('item_id') 
#             logger.info(f"Text Item num: {len(self.env)}")
#         else:
#             # 这里吃 cpu 很多
#             self.env = self.env.set_index('item_id').T.to_dict()
#         logger.info(f"Text Item num: {len(self.env)}")

#     def _neg_sample(self, item_set):
#         item = random.randint(1, self.item_num - 1)
#         while item in item_set:
#             item = random.randint(1, self.item_num - 1)
#         return item

#     def _padding_sequence(self, sequence, max_length, random_sample=False):
#         pad_len = max_length - len(sequence)
#         if random_sample:
#             pad_seq = [self._neg_sample(sequence) for _ in range(pad_len)]
#             sequence = pad_seq + sequence
#         else:
#             sequence = [0] * pad_len + sequence
#         sequence = sequence[-max_length:]
#         return torch.tensor(sequence, dtype=torch.long)

#     def reconstruct_train_data(self, item_seq):
#         masked_index = []
#         neg_item = []
#         item_seq_len = len(item_seq)
#         for i in range(item_seq_len - 1):
#             neg_item.append(self._neg_sample(item_seq))
#             masked_index.append(1)

#         item_seq = self._padding_sequence(list(item_seq), self.max_seq_length, random_sample=self.random_sample)
#         masked_index = self._padding_sequence(masked_index, self.max_seq_length-1)
#         if self.num_negatives:
#             neg_item = []
#             for _ in range(self.num_negatives):
#                 neg_item.append(self._neg_sample([]))
#         else:
#             neg_item = self._padding_sequence(neg_item, self.max_seq_length, random_sample=self.random_sample)
#         return item_seq, neg_item, masked_index

#     def _padding_time_sequence(self, sequence, max_length):
#         pad_len = max_length - len(sequence)
#         sequence = [0] * pad_len + sequence
#         sequence = sequence[-max_length:]
#         vq_time = []
#         for time in sequence:
#             dt = datetime.datetime.fromtimestamp(time, pytz.timezone('UTC'))
#             vq_time.append([dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second])
#         return torch.tensor(vq_time, dtype=torch.long)

#     def __getitem__(self, index):

#         item_seq = self.train_seq[index]
#         item_seq, neg_item, masked_index = self.reconstruct_train_data(item_seq)
#         time_seq = self.train_time_seq[index]
#         time_seq = self._padding_time_sequence(list(time_seq), self.max_seq_length)
#         # 物品ID转换为token
#         item_seq_token = self.id2token[item_seq]
#         neg_items_token = self.id2token[neg_item]
#         pos_input_ids, pos_cu_input_lens, pos_position_ids = [], [], []
#         neg_input_ids, neg_cu_input_lens, neg_position_ids = [], [], []

#         # 文本特征处理函数
#         def process_item(item):
#             if item != self.id2token[0] and item not in self.env:
#                 # assert item in self.env, f"{item}"
#                 logger.info(f"{item} not in self.env")
            
#             if self.config['memory_optimize']:
#                 try:
#                     item_i = self.env.loc[item]  # 使用 Pandas 索引查找
#                 except KeyError:
#                     item_i = {}
#             else:
#                 item_i = self.env.get(item, {})
#             text_str = ""
#             if len(item_i):
#                 text_str = f"{self.item_prompt}"
#                 for key in self.text_keys:
#                     value = item_i[key]
#                     if value and str(value) != 'nan':
#                         text_str += f"{key}: {value}"

#             # ids = self.tokenizer.encode(text_str)
#             # ids = ids[:self.max_text_length]
#             ids = self.tokenizer.encode(
#                 text_str,
#                 truncation=True,
#                 max_length=self.max_text_length,
#                 add_special_tokens=True
#             )
#             mask = [1] * len(ids) # 创建掩码
#             return ids, mask

#         for item in item_seq_token:
#             ids, _ = process_item(item)
#             pos_input_ids.extend(ids + [0] * self.item_emb_token_n)
#             pos_cu_input_lens.append(len(ids) + self.item_emb_token_n)
#             # 生成位置ID
#             pos_position_ids.extend((torch.arange(len(ids) + self.item_emb_token_n) + (self.max_text_length - len(ids))).tolist())

#         for neg in neg_items_token:
#             ids, _ = process_item(neg)
#             neg_input_ids.extend(ids + [0] * self.item_emb_token_n)
#             neg_cu_input_lens.append(len(ids) + self.item_emb_token_n)
#             neg_position_ids.extend((torch.arange(len(ids) + self.item_emb_token_n) + (self.max_text_length - len(ids))).tolist())


#         current_sample_len = len(pos_input_ids)
    
#         # 为了避免几十万条日志刷屏，我们只打印前 5 个索引，或者异常大的样本
#         if index < 5 or current_sample_len > 3000: 
#             # 获取当前显存占用 (MB)
#             mem_alloc = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
#             print(f"")
#             print(f"--- [Item Monitor] Index: {index} ---")
#             print(f"Item Seq Len: {len(item_seq)}")   # 原始物品序列长度 (比如 50)
#             print(f"Text Token Len: {current_sample_len}") # 膨胀后的文本 Token 长度 (比如 3250)
#             print(f"Current GPU Mem: {mem_alloc:.2f} MB")
#             print(f"-------------------------------------")
#         # ================== 【新增监控代码结束】 ==================

#         outputs = {
#             "pos_item_ids": torch.as_tensor(item_seq, dtype=torch.int64),  # 正样本物品ID
#             "neg_item_ids": torch.as_tensor(neg_item, dtype=torch.int64),  # 负样本物品ID
#             "pos_input_ids": torch.as_tensor(pos_input_ids, dtype=torch.int64),  # 正样本文本ID
#             "pos_cu_input_lens": torch.as_tensor(pos_cu_input_lens, dtype=torch.int64),  # 正样本累积长度
#             "pos_position_ids": torch.as_tensor(pos_position_ids, dtype=torch.int64),  # 正样本位置ID
#             "neg_input_ids": torch.as_tensor(neg_input_ids, dtype=torch.int64),  # 负样本文本ID
#             "neg_cu_input_lens": torch.as_tensor(neg_cu_input_lens, dtype=torch.int64),  # 负样本累积长度
#             "neg_position_ids": torch.as_tensor(neg_position_ids, dtype=torch.int64),  # 负样本位置ID
#             "attention_mask": torch.as_tensor(masked_index, dtype=torch.int64),  # 注意力掩码
#             "time_ids": torch.as_tensor(time_seq, dtype=torch.int64),  # 时间特征
#         }
#         return outputs

class TextSEQTrainDataset(Dataset):
    def __init__(self, config, dataload):
        self.dataload = dataload
        self.config = config

        self.item_num = dataload.item_num
        self.train_seq = dataload.train_feat['item_seq']
        self.length = len(self.train_seq)
        self.train_time_seq = dataload.train_feat['time_seq']
        self.id2token = dataload.id2token['item_id']

        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']+1
        self.max_text_length = config['MAX_TEXT_LENGTH']
        self.device = config['device']

        self.text_path = config['text_path']
        self.text_keys = config['text_keys']
        self.tokenizer = AutoTokenizer.from_pretrained(config['item_pretrain_dir'], trust_remote_code=True)
        self.item_prompt = config['item_prompt']
        self.item_emb_token_n = config['item_emb_token_n']
        self.num_negatives = config['num_negatives']
        self.random_sample = True if config['loss'] and config['loss'] == 'nce' else False
        if self.num_negatives:
            self.num_negatives = math.ceil(self.num_negatives / dist.get_world_size() / config['train_batch_size'])
        
        logger.info(f"Text path: {self.text_path}")
        self.load_content()

    def __len__(self):
        return self.length

    def load_content(self):
        # === 核心修改：使用 HuggingFace Datasets 进行内存映射读取 ===
        logger.info("Loading dataset with memory mapping (HuggingFace Datasets)...")
        # 直接读取 CSV，不加载进 RAM，而是映射到硬盘
        self.hf_dataset = load_dataset('csv', data_files=self.text_path, split='train')
        
        # 建立 item_id -> row_index 的轻量级索引
        # 假设 item_id 是唯一的字符串
        self.item_id_to_idx = {str(item_id): idx for idx, item_id in enumerate(self.hf_dataset['item_id'])}
        logger.info(f"Dataset mapped. Total rows: {len(self.hf_dataset)}")

    def _neg_sample(self, item_set):
        item = random.randint(1, self.item_num - 1)
        while item in item_set:
            item = random.randint(1, self.item_num - 1)
        return item

    def _padding_sequence(self, sequence, max_length, random_sample=False):
        pad_len = max_length - len(sequence)
        if random_sample:
            pad_seq = [self._neg_sample(sequence) for _ in range(pad_len)]
            sequence = pad_seq + sequence
        else:
            sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:]
        return torch.tensor(sequence, dtype=torch.long)

    def reconstruct_train_data(self, item_seq):
        masked_index = []
        neg_item = []
        item_seq_len = len(item_seq)
        for i in range(item_seq_len - 1):
            neg_item.append(self._neg_sample(item_seq))
            masked_index.append(1)

        item_seq = self._padding_sequence(list(item_seq), self.max_seq_length, random_sample=self.random_sample)
        masked_index = self._padding_sequence(masked_index, self.max_seq_length-1)
        if self.num_negatives:
            neg_item = []
            for _ in range(self.num_negatives):
                neg_item.append(self._neg_sample([]))
        else:
            neg_item = self._padding_sequence(neg_item, self.max_seq_length, random_sample=self.random_sample)
        return item_seq, neg_item, masked_index

    def _padding_time_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:]
        vq_time = []
        for time in sequence:
            dt = datetime.datetime.fromtimestamp(time, pytz.timezone('UTC'))
            vq_time.append([dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second])
        return torch.tensor(vq_time, dtype=torch.long)

    def __getitem__(self, index):
        item_seq = self.train_seq[index]
        item_seq, neg_item, masked_index = self.reconstruct_train_data(item_seq)
        time_seq = self.train_time_seq[index]
        time_seq = self._padding_time_sequence(list(time_seq), self.max_seq_length)
        
        item_seq_token = self.id2token[item_seq]
        neg_items_token = self.id2token[neg_item]
        pos_input_ids, pos_cu_input_lens, pos_position_ids = [], [], []
        neg_input_ids, neg_cu_input_lens, neg_position_ids = [], [], []

        def process_item(item):
            # === 修改：从 Disk Dataset 读取 ===
            item_str = str(item)
            item_data = {}
            if item_str in self.item_id_to_idx:
                idx = self.item_id_to_idx[item_str]
                item_data = self.hf_dataset[idx] # 此时才从硬盘读取
            else:
                if item != self.id2token[0]:
                    logger.info(f"{item} not in dataset")

            text_str = ""
            if item_data:
                text_str = f"{self.item_prompt}"
                for key in self.text_keys:
                    value = item_data.get(key)
                    if value is not None and str(value) != 'nan' and str(value) != '':
                        text_str += f"{key}: {value}"

            ids = self.tokenizer.encode(
                text_str,
                truncation=True,
                max_length=self.max_text_length,
                add_special_tokens=True
            )
            mask = [1] * len(ids)
            return ids, mask

        for item in item_seq_token:
            ids, _ = process_item(item)
            pos_input_ids.extend(ids + [0] * self.item_emb_token_n)
            pos_cu_input_lens.append(len(ids) + self.item_emb_token_n)
            pos_position_ids.extend((torch.arange(len(ids) + self.item_emb_token_n) + (self.max_text_length - len(ids))).tolist())

        for neg in neg_items_token:
            ids, _ = process_item(neg)
            neg_input_ids.extend(ids + [0] * self.item_emb_token_n)
            neg_cu_input_lens.append(len(ids) + self.item_emb_token_n)
            neg_position_ids.extend((torch.arange(len(ids) + self.item_emb_token_n) + (self.max_text_length - len(ids))).tolist())

        outputs = {
            "pos_item_ids": torch.as_tensor(item_seq, dtype=torch.int64),
            "neg_item_ids": torch.as_tensor(neg_item, dtype=torch.int64),
            "pos_input_ids": torch.as_tensor(pos_input_ids, dtype=torch.int64),
            "pos_cu_input_lens": torch.as_tensor(pos_cu_input_lens, dtype=torch.int64),
            "pos_position_ids": torch.as_tensor(pos_position_ids, dtype=torch.int64),
            "neg_input_ids": torch.as_tensor(neg_input_ids, dtype=torch.int64),
            "neg_cu_input_lens": torch.as_tensor(neg_cu_input_lens, dtype=torch.int64),
            "neg_position_ids": torch.as_tensor(neg_position_ids, dtype=torch.int64),
            "attention_mask": torch.as_tensor(masked_index, dtype=torch.int64),
            "time_ids": torch.as_tensor(time_seq, dtype=torch.int64),
        }
        return outputs