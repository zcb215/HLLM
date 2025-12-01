# Copyright (c) 2024 westlake-repl
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
# SPDX-License-Identifier: MIT
# This file has been modified by Junyi Chen.
#
# Original file was released under MIT, with the full license text
# available at https://choosealicense.com/licenses/mit/.
#
# This modified file is released under the same license.

import copy
import pickle
import os
import yaml
from collections import Counter
from logging import getLogger

import numpy as np
import pandas as pd
import torch

from REC.utils import set_color
from REC.utils.enum_type import InputType
from torch_geometric.utils import degree


class Data:
    def __init__(self, config):
        self.config = config
        # print(f"Data config:{config}")
        # input()
        self.dataset_path = config['data_path']
        self.dataset_name = config['dataset']
        # print(f"config['dataset']:{config['dataset']}")
        # input()
        self.data_split = config['data_split']
        # print(f"config['data_split']:{config['data_split']}")   # non
        self.item_data = config['item_data']
        # print(f"config['item_data']:{config['item_data']}")   #non
        # input()
        self.logger = getLogger()
        self._from_scratch()

    def _from_scratch(self):
        self.logger.info(set_color(f'Loading {self.__class__} from scratch with {self.data_split = }.', 'green'))
        self._load_inter_feat(self.dataset_name, self.dataset_path, self.item_data)
        self._data_processing()

    def _load_inter_feat(self, token, dataset_path, item_data=None):
        inter_feat_path = os.path.join(dataset_path, f'{token}.csv')
        if not os.path.isfile(inter_feat_path):
            raise ValueError(f'File {inter_feat_path} not exist.')

        df = pd.read_csv(
            inter_feat_path, delimiter=',', dtype={'item_id': str, 'user_id': str, 'timestamp': int}, header=0, names=['item_id', 'user_id', 'timestamp']
        )
        self.logger.info(f'Interaction feature loaded successfully from [{inter_feat_path}].')
        self.inter_feat = df

        if item_data:
            item_data_path = os.path.join(dataset_path, f'{item_data}.csv')
            item_df = pd.read_csv(
                item_data_path, delimiter=',', dtype={'item_id': str, 'user_id': str, 'timestamp': int}, header=0, names=['item_id', 'user_id', 'timestamp']
            )
            self.item_feat = item_df
            self.logger.info(f'Item feature loaded successfully from [{item_data}].')

    def _data_processing(self):

        self.id2token = {}  # 索引 -> 原始ID
        self.token2id = {}  # 原始ID -> 索引
        remap_list = ['user_id', 'item_id']
        for feature in remap_list:
            if feature == 'item_id' and self.item_data:
                feats = self.item_feat[feature]  # 来源 A：物品详情表
                feats_raw = self.inter_feat[feature]
            else:
                feats = self.inter_feat[feature]
            # print(f"self.inter_feat:\n {self.inter_feat}")
            # print(f"feats:\n {feats}")
            # input()
            # 相当于离散化，讲原先的id转化为数字信息，同时map则是记录原始数据的，可以使用前面的 idlist和mp还原回原始的数据信息
            # mp: 唯一值数组，即原始数据中去重后的值， new_ids_list  则是全数据的 id 表示
            new_ids_list, mp = pd.factorize(feats)
            # print(f"new_ids_list:\n {new_ids_list}")
            # print(f"mp:\n {mp}")
            # input()
            # 2. 添加 [PAD] 标记
            mp = ['[PAD]'] + list(mp)
            # mp: ['[PAD]', 'user3', 'user1', 'user2']
            # print(f"mp:\n {mp}")
            # input()
            token_id = {t: i for i, t in enumerate(mp)}
            # 下面是补充存在物品集合里面但是没有交互过的物品 id
            if feature == 'item_id' and self.item_data:
                _, raw_mp = pd.factorize(feats_raw)
                for x in raw_mp:
                    if x not in token_id:
                        token_id[x] = len(token_id)
                        mp.append(x)
            mp = np.array(mp)

            self.id2token[feature] = mp
            self.token2id[feature] = token_id
            #应用 id 
            self.inter_feat[feature] = self.inter_feat[feature].map(token_id)

        self.user_num = len(self.id2token['user_id'])
        self.item_num = len(self.id2token['item_id'])
        self.logger.info(f"{self.user_num = } {self.item_num = }")
        self.logger.info(f"{self.inter_feat['item_id'].isna().any() = } {self.inter_feat['user_id'].isna().any() = }")
        self.inter_num = len(self.inter_feat)
        self.uid_field = 'user_id'
        self.iid_field = 'item_id'
        self.user_seq = None
        self.train_feat = None
        self.feat_name_list = ['inter_feat']  # self.inter_feat

    def build(self):
        self.logger.info(f"build {self.dataset_name} dataload")
        self.sort(by='timestamp')
        # print(self.inter_feat.head(80))
        # input()
        #         item_id  user_id  timestamp
        # 5911050   165053   409678  832550400
        # 186647     51599    13186  835660800
        # 186648     24310    13186  840240000
        # 186649      9828    13186  843004800
        # 8733632   578448   604234  848016000
        # ...          ...      ...        ...
        # 487945    144107    33356  866505600
        # 5332502    39863   368729  866505600
        # 487944    207202    33356  866505600
        # 487943    208405    33356  866505600
        # 186654      4135    13186  866592000
        user_list = self.inter_feat['user_id'].values
        item_list = self.inter_feat['item_id'].values
        timestamp_list = self.inter_feat['timestamp'].values
        grouped_index = self._grouped_index(user_list)

        user_seq = {}
        time_seq = {}
        for uid, index in grouped_index.items():
            user_seq[uid] = item_list[index]
            time_seq[uid] = timestamp_list[index]

        # 依据用户的进行分组，得到基于用户的交互记录，依次是交互的物品和交互时间
        self.user_seq = user_seq
        self.time_seq = time_seq
        train_feat = dict()
        indices = []

        # 对于训练数据进行调整，删除最后两列
        for index in grouped_index.values():
            indices.extend(list(index)[:-2])
        
        # print(f"indices :{indices[:20]}")
        # input()
        # 这里 的 train_feat 数据会按照indices 的顺序写入  而indices  又是按照用户分类的
        # 所有 train_feat 最终的效果是 先按照用户分类 在按照 时间排序
            
        # item_id
        # user_id
        # timestamp
        for k in self.inter_feat:
            # print(k)
            train_feat[k] = self.inter_feat[k].values[indices]
        # input()
        if self.config['MODEL_INPUT_TYPE'] == InputType.AUGSEQ:
            train_feat = self._build_aug_seq(train_feat)
        elif self.config['MODEL_INPUT_TYPE'] == InputType.SEQ:
            train_feat = self._build_seq(train_feat)
        # print(f"seq_train_feat:{train_feat}")
        # input()
        self.train_feat = train_feat

    #用于分组操作，以便可以根据键快速找到所有出现的位置。
    def _grouped_index(self, group_by_list):
        index = {}
        for i, key in enumerate(group_by_list):
            if key not in index:
                index[key] = [i]
            else:
                index[key].append(i)
        return index

    def _build_seq(self, train_feat):
        max_item_list_len = self.config['MAX_ITEM_LIST_LENGTH']+1

        uid_list, item_list_index = [], []
        seq_start = 0
        save = False
        user_list = train_feat['user_id']
        user_list = np.append(user_list, -1)
        last_uid = user_list[0]
        for i, uid in enumerate(user_list):
            if last_uid != uid:
                save = True
            if save:
                if (self.data_split is None or self.data_split == True) and i - seq_start > max_item_list_len:
                    offset = (i - seq_start) % max_item_list_len
                    # 去除太古早的历史部分
                    seq_start += offset
                    x = torch.arange(seq_start, i)
                    sx = torch.split(x, max_item_list_len)
                    for sub in sx:
                        uid_list.append(last_uid)
                        # 这里存入的是物品的索引
                        item_list_index.append(slice(sub[0], sub[-1]+1))
                else:
                    uid_list.append(last_uid)
                    item_list_index.append(slice(seq_start, i))  # maybe too long but will be truncated in dataloader

                save = False
                last_uid = uid
                seq_start = i

        seq_train_feat = {}
        seq_train_feat['user_id'] = np.array(uid_list)
        seq_train_feat['item_seq'] = []
        seq_train_feat['time_seq'] = []
        # 从 物品集 的下标 转为具体的数据     这里的是砍掉了最后两个的数据
        for index in item_list_index:
            # train_feat['item_id'][index] 等同于 [101, 102, 103, 201...][0:3]
            seq_train_feat['item_seq'].append(train_feat['item_id'][index])
            seq_train_feat['time_seq'].append(train_feat['timestamp'][index])

        return seq_train_feat

    # 自回归（Next Item Prediction）数据增强方式
    def _build_aug_seq(self, train_feat):
        max_item_list_len = self.config['MAX_ITEM_LIST_LENGTH']+1

        # by = ['user_id', 'timestamp']
        # ascending = [True, True]
        # for b, a in zip(by[::-1], ascending[::-1]):
        #     index = np.argsort(train_feat[b], kind='stable')
        #     if not a:
        #         index = index[::-1]
        #     for k in train_feat:
        #         train_feat[k] = train_feat[k][index]

        uid_list, item_list_index = [], []
        seq_start = 0
        save = False
        user_list = train_feat['user_id']
        user_list = np.append(user_list, -1)
        last_uid = user_list[0]
        for i, uid in enumerate(user_list):
            if last_uid != uid:
                save = True
            if save:
                if i - seq_start > max_item_list_len:
                    offset = (i - seq_start) % max_item_list_len
                    seq_start += offset
                    x = torch.arange(seq_start, i)
                    sx = torch.split(x, max_item_list_len)
                    for sub in sx:
                        uid_list.append(last_uid)
                        item_list_index.append(slice(sub[0], sub[-1]+1))
                else:
                    uid_list.append(last_uid)
                    item_list_index.append(slice(seq_start, i))
                save = False
                last_uid = uid
                seq_start = i

        seq_train_feat = {}
        aug_uid_list = []
        aug_item_list = []
        for uid, item_index in zip(uid_list, item_list_index):
            # 获取的是切片的 start 和 end
            st = item_index.start
            ed = item_index.stop
            # 增强 序列  比如长度为2的序列 最终产生的输出如下
            # 用户B-子序列1:
            # - [item7]                          # 长度1
            # - [item7, item8]                   # 长度2
            lens = ed - st
            for sub_idx in range(1, lens):
                aug_item_list.append(train_feat['item_id'][slice(st, st+sub_idx+1)])
                aug_uid_list.append(uid)

        seq_train_feat['user_id'] = np.array(aug_uid_list)
        seq_train_feat['item_seq'] = aug_item_list

        return seq_train_feat

    def sort(self, by, ascending=True):

        if isinstance(self.inter_feat, pd.DataFrame):
            self.inter_feat.sort_values(by=by, ascending=ascending, inplace=True)

        else:
            if isinstance(by, str):
                by = [by]

            if isinstance(ascending, bool):
                ascending = [ascending]

            if len(by) != len(ascending):
                if len(ascending) == 1:
                    ascending = ascending * len(by)
                else:
                    raise ValueError(f'by [{by}] and ascending [{ascending}] should have same length.')
            for b, a in zip(by[::-1], ascending[::-1]):
                index = np.argsort(self.inter_feat[b], kind='stable')
                if not a:
                    index = index[::-1]
                for k in self.inter_feat:
                    self.inter_feat[k] = self.inter_feat[k][index]

    @property
    def avg_actions_of_users(self):
        # 计算每个用户平均有多少条交互记录
        """Get the average number of users' interaction records.

        Returns:
            numpy.float64: Average number of users' interaction records.
        """
        if isinstance(self.inter_feat, pd.DataFrame):
            return np.mean(self.inter_feat.groupby(self.uid_field).size())
        else:
            return np.mean(list(Counter(self.inter_feat[self.uid_field]).values()))

    @property
    def avg_actions_of_items(self):
        """Get the average number of items' interaction records.

        Returns:
            numpy.float64: Average number of items' interaction records.
        """
        if isinstance(self.inter_feat, pd.DataFrame):
            return np.mean(self.inter_feat.groupby(self.iid_field).size())
        else:
            return np.mean(list(Counter(self.inter_feat[self.iid_field]).values()))

    @property
    def sparsity(self):
        """Get the sparsity of this dataset.

        Returns:
            float: Sparsity of this dataset.
        """
        return 1 - self.inter_num / self.user_num / self.item_num

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        info = [set_color(self.dataset_name, 'pink')]
        if self.uid_field:
            info.extend([
                set_color('The number of users', 'blue') + f': {self.user_num}',
                set_color('Average actions of users', 'blue') + f': {self.avg_actions_of_users}'
            ])
        if self.iid_field:
            info.extend([
                set_color('The number of items', 'blue') + f': {self.item_num}',
                set_color('Average actions of items', 'blue') + f': {self.avg_actions_of_items}'
            ])
        info.append(set_color('The number of inters', 'blue') + f': {self.inter_num}')
        if self.uid_field and self.iid_field:
            info.append(set_color('The sparsity of the dataset', 'blue') + f': {self.sparsity * 100}%')

        return '\n'.join(info)

    def copy(self, new_inter_feat):
        """Given a new interaction feature, return a new :class:`Dataset` object,
        whose interaction feature is updated with ``new_inter_feat``, and all the other attributes the same.

        Args:
            new_inter_feat (Interaction): The new interaction feature need to be updated.

        Returns:
            :class:`~Dataset`: the new :class:`~Dataset` object, whose interaction feature has been updated.
        """
        nxt = copy.copy(self)
        nxt.inter_feat = new_inter_feat
        return nxt

    def counter(self, field):
        if isinstance(self.inter_feat, pd.DataFrame):
            return Counter(self.inter_feat[field].values)
        else:
            return Counter(self.inter_feat[field])

    @property
    def user_counter(self):
        return self.counter('user_id')

    @property
    def item_counter(self):
        return self.counter('item_id')

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.
        Construct the square matrix from the training data and normalize it
        using the laplace matrix.
        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}
        Returns:
            The normalized interaction matrix in Tensor.
        """

        row = torch.tensor(self.train_feat[self.uid_field])
        col = torch.tensor(self.train_feat[self.iid_field]) + self.user_num
        edge_index1 = torch.stack([row, col])
        edge_index2 = torch.stack([col, row])
        edge_index = torch.cat([edge_index1, edge_index2], dim=1)

        deg = degree(edge_index[0], self.user_num + self.item_num)

        norm_deg = 1. / torch.sqrt(torch.where(deg == 0, torch.ones([1]), deg))
        edge_weight = norm_deg[edge_index[0]] * norm_deg[edge_index[1]]

        return edge_index, edge_weight
