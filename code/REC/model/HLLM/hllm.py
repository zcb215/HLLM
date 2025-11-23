# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import transformers
from transformers import AutoConfig, AutoModelForCausalLM
from logging import getLogger

from REC.utils.enum_type import InputType
from REC.model.basemodel import BaseModel, all_gather
from REC.model.HLLM.modeling_llama import LlamaForCausalLM
from REC.model.HLLM.modeling_mistral import MistralForCausalLM
from REC.model.HLLM.modeling_bert import BertModel
from REC.model.HLLM.baichuan.modeling_baichuan import BaichuanForCausalLM


class HLLM(BaseModel):
    # 设定输入数据的格式
    input_type = InputType.SEQ

    def __init__(self, config, dataload):
        super(HLLM, self).__init__()
        self.logger = getLogger()

        self.item_pretrain_dir = config['item_pretrain_dir']
        self.user_pretrain_dir = config['user_pretrain_dir']
        self.gradient_checkpointing = config['gradient_checkpointing']
        self.use_ft_flash_attn = config['use_ft_flash_attn']

        # 依据论文里面的架构，设计物品模型和用户模型
        self.logger.info(f"create item llm")
        self.item_llm = self.create_llm(self.item_pretrain_dir, config['item_llm_init'])
        self.logger.info(f"create user llm")
        self.user_llm = self.create_llm(self.user_pretrain_dir, config['user_llm_init'])

        # 可学习的物品嵌入token
        self.item_emb_token_n = config['item_emb_token_n']
        if self.item_emb_token_n > 1:
            raise NotImplementedError(f"Not support item_emb_token_n {self.item_emb_token_n} > 1")

        if self.item_emb_token_n > 0:
            self.item_emb_tokens = nn.Parameter(
                torch.zeros(1, self.item_emb_token_n, self.item_llm.config.hidden_size)
            )
            self.item_emb_tokens.data.normal_(mean=0.0, std=0.02)
            # 加载预训练的物品嵌入令牌权重
            if config['item_emb_pretrain']:
                ckpt = torch.load(config['item_emb_pretrain'], map_location='cpu')
                self.logger.info(f"load item_emb_token from {config['item_emb_pretrain']} with {ckpt.size()}")
                self.item_emb_tokens.data = nn.Parameter(ckpt)
        else:  # mean pooling
            self.item_emb_tokens = None

        self.loss = config['loss']
        if self.loss == 'nce':
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.nce_thres = config['nce_thres'] if config['nce_thres'] else 0.99
            self.num_negatives = config['num_negatives']
            self.logger.info(f"nce thres setting to {self.nce_thres}")
        else:
            raise NotImplementedError(f"Only nce is supported")

        if config['load_pretrain']:
            state_dict = torch.load(config['load_pretrain'], map_location="cpu")
            msg = self.load_state_dict(state_dict, strict=False)
            self.logger.info(f"{msg.missing_keys = }")
            self.logger.info(f"{msg.unexpected_keys = }")

    # LLM 创建方法  根据配置创建不同类型的LLM
    def create_llm(self, pretrain_dir, init=True):
        self.logger.info(f"******* create LLM {pretrain_dir} *******")

        # 从预训练目录加载模型配置  
        hf_config = AutoConfig.from_pretrained(pretrain_dir, trust_remote_code=True)
        # todo   可以输出一下 hf config的配置信息
        self.logger.info(f"hf_config: {hf_config}")
        hf_config.gradient_checkpointing = self.gradient_checkpointing  # 梯度检查点节省显存
        hf_config.use_cache = False           # 禁用KV缓存，节省推理内存
        hf_config.output_hidden_states = True # 输出所有隐藏状态，用于推荐任务
        hf_config.return_dict = True          # 返回字典格式，便于访问不同输出

        self.logger.info("xxxxx starting loading checkpoint")

        # Llama模型 
        #   纯Decoder架构，自回归生成
        # - RoPE位置编码，支持长序列
        # - 使用SwiGLU激活函数
        # - 预训练数据量大，通用性强
        if isinstance(hf_config, transformers.LlamaConfig):
            hf_config.use_ft_flash_attn = self.use_ft_flash_attn
            self.logger.info(f'Using flash attention {hf_config.use_ft_flash_attn} for llama')
            self.logger.info(f'Init {init} for llama')
            if init:
                return LlamaForCausalLM.from_pretrained(pretrain_dir, config=hf_config)
            else:
                return LlamaForCausalLM(config=hf_config).cuda()
        # Mistral模型
        # - Sliding Window Attention (SWA)
        # - 更高效的注意力计算
        # - 较小的参数量，较强的性能
        # - 优化的KV缓存机制
        elif isinstance(hf_config, transformers.MistralConfig):
            hf_config.use_ft_flash_attn = self.use_ft_flash_attn
            self.logger.info(f'Using flash attention {hf_config.use_ft_flash_attn} for mistral')
            self.logger.info(f'Init {init} for mistral')
            if init:
                return MistralForCausalLM.from_pretrained(pretrain_dir, config=hf_config)
            else:
                return MistralForCausalLM(config=hf_config).cuda()
            
        # BERT模型
        elif isinstance(hf_config, transformers.BertConfig):
            hf_config.use_ft_flash_attn = self.use_ft_flash_attn
            self.logger.info(f'Using flash attention {hf_config.use_ft_flash_attn} for bert')
            self.logger.info(f'Init {init} for bert')
            if init:
                return BertModel.from_pretrained(pretrain_dir, config=hf_config)
            else:
                return BertModel(config=hf_config).cuda()
            
        # Baichuan模型
        # - 专门针对中文训练
        # - 理解中文语言特性
        # - 支持中文多轮对话
        # - 包含中文知识图谱
        elif getattr(hf_config, "model_type", None) == "baichuan":
            hf_config.use_ft_flash_attn = self.use_ft_flash_attn
            self.logger.info(f'Using flash attention {hf_config.use_ft_flash_attn} for baichuan')
            self.logger.info(f'Init {init} for baichuan')
            if init:
                return BaichuanForCausalLM.from_pretrained(pretrain_dir, config=hf_config)
            else:
                return BaichuanForCausalLM(config=hf_config).cuda()
        else:
            return AutoModelForCausalLM.from_pretrained(
                self.local_dir, config=hf_config
            )

    # 噪声对比估计（NCE）损失函数
    def nce_loss(self, cur_embs, target_pos, target_neg, user_attention_mask):
        # logit_scale 是温度参数的逆，控制分布的尖锐程度
        with torch.no_grad():
            self.logit_scale.clamp_(0, np.log(100))
        logit_scale = self.logit_scale.exp()
        D = target_neg.size(-1)
        # 所有向量进行L2归一化，转换为单位向量
        output_embs = cur_embs / cur_embs.norm(dim=-1, keepdim=True)
        target_pos_embs = target_pos / target_pos.norm(dim=-1, keepdim=True)
        # 计算当前表示与正样本的余弦相似度
        pos_logits = F.cosine_similarity(output_embs, target_pos_embs, dim=-1).unsqueeze(-1)

        target_neg = target_neg / target_neg.norm(dim=-1, keepdim=True)

        neg_embedding_all = all_gather(target_neg, sync_grads=True).reshape(-1, D)  # [num, dim]
        neg_embedding_all = neg_embedding_all.transpose(-1, -2)

        # 计算当前样本与所有负样本的相似度
        neg_logits = torch.matmul(output_embs, neg_embedding_all)
        # 计算目标正样本与所有负样本的相似度
        fix_logits = torch.matmul(target_pos_embs, neg_embedding_all)
        # 过于相似的负样本的logits设置为最小值   防止过拟合
        neg_logits[fix_logits > self.nce_thres] = torch.finfo(neg_logits.dtype).min

        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        logits = logits[user_attention_mask.bool()] * logit_scale
        # 对比学习里面第 一 个是正样本  这里当中多分类问题
        labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.int64)
        return logits, labels

    # 基于LLM的物品嵌入生成函数
    def forward_item_emb(
        self,
        input_ids,        # 输入的token IDs，形状: (总token数,)
        position_ids,      # 位置编码IDs
        cu_input_lens,     # 每个序列的长度，用于累积计算
        emb_token_n,       # 特殊嵌入token的数量
        emb_tokens,        # 特殊嵌入token的向量
        llm                # 语言模型实例
    ):
        
        # llm.get_input_embeddings()作用：获取语言模型的词嵌入层（embedding layer）
        inputs_embeds = llm.get_input_embeddings()(input_ids)
        # 计算每个序列的结束位置  也就是计算相对于总体的结束位置
        emb_pos = cu_input_lens.cumsum(dim=0, dtype=torch.int32)
        if emb_token_n > 0:
            # emb_pos - 1 表示每个序列的最后一个token的位置

            # todo  这里用emb_tokens覆盖了原始序列最后一个token的嵌入，这确实会丢失原始信息
            inputs_embeds[emb_pos - 1] = emb_tokens

        # 将输入传递给LLM，获取最后一层的隐藏状态
        model_out = llm(inputs_embeds=inputs_embeds.unsqueeze(0), cu_input_lens=cu_input_lens, position_ids=position_ids.unsqueeze(0))
        model_out = model_out.hidden_states[-1].squeeze(0)

        if emb_token_n > 0:
            emb = model_out[emb_pos - 1]
        else:
            max_len = cu_input_lens.max().item()
            cu_seqlens = F.pad(cu_input_lens.cumsum(dim=0, dtype=torch.int32), (1, 0))
            seqs = [model_out[start:end] for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])]
            padded_seqs = [
                F.pad(
                    seqs[i],
                    (0, 0) * (seqs[i].dim() - 1) + (0, max_len - cu_input_lens[i]),
                    value=0.0,
                )
                for i in range(cu_input_lens.size(0))
            ]
            out = torch.stack(padded_seqs)
            emb = out.sum(dim=1) / cu_input_lens.unsqueeze(1)

        return emb

    def forward(self, interaction, mode='train'):
        if mode == 'predict':
            return self.predict(interaction[0], interaction[1], interaction[2])
        if mode == 'compute_item':
            return self.compute_item(interaction)
        user_attention_mask = interaction['attention_mask']
        N, S = user_attention_mask.shape
        pos_input_ids, pos_cu_input_lens, pos_position_ids = interaction['pos_input_ids'], interaction['pos_cu_input_lens'], interaction['pos_position_ids']
        neg_input_ids, neg_cu_input_lens, neg_position_ids = interaction['neg_input_ids'], interaction['neg_cu_input_lens'], interaction['neg_position_ids']

        pos_embedding = self.forward_item_emb(pos_input_ids, pos_position_ids, pos_cu_input_lens, self.item_emb_token_n, self.item_emb_tokens, self.item_llm)
        pos_embedding = pos_embedding.reshape(N, S+1, -1)
        neg_embedding = self.forward_item_emb(neg_input_ids, neg_position_ids, neg_cu_input_lens, self.item_emb_token_n, self.item_emb_tokens, self.item_llm)
        neg_embedding = neg_embedding.reshape(N, -1, self.item_llm.config.hidden_size)

        target_pos_embs = pos_embedding[:, 1:]
        target_neg_embs = neg_embedding

        user_embedding = self.user_llm(inputs_embeds=pos_embedding[:, :-1], attention_mask=user_attention_mask).hidden_states[-1]

        model_out = {}
        logits, labels = self.nce_loss(user_embedding, target_pos_embs, target_neg_embs, user_attention_mask)
        model_out['loss'] = F.cross_entropy(logits, labels)
        model_out['nce_samples'] = (logits > torch.finfo(logits.dtype).min/100).sum(dim=1).float().mean()  # samples after filtering same negatives
        for k in [1, 5, 10, 50, 100]:
            if k > logits.size(1):
                break
            indices = logits.topk(k, dim=1).indices
            model_out[f"nce_top{k}_acc"] = labels.view(-1, 1).eq(indices).any(dim=1).float().mean()
        return model_out

    @torch.no_grad()
    def predict(self, item_seq, time_seq, item_feature):
        attention_mask = (item_seq > 0).int()

        pos_embedding = item_feature[item_seq]

        user_embedding = self.user_llm(inputs_embeds=pos_embedding, attention_mask=attention_mask).hidden_states[-1]
        seq_output = user_embedding[:, -1]
        seq_output = seq_output / seq_output.norm(dim=-1, keepdim=True)
        item_feature = item_feature / item_feature.norm(dim=-1, keepdim=True)

        return torch.matmul(seq_output, item_feature.t())

    @torch.no_grad()
    def compute_item_all(self):
        return self.item_embedding.weight

    @torch.no_grad()
    def compute_item(self, interaction):
        pos_input_ids, pos_cu_input_lens, pos_position_ids = interaction['pos_input_ids'], interaction['pos_cu_input_lens'], interaction['pos_position_ids']
        pos_embedding = self.forward_item_emb(pos_input_ids, pos_position_ids, pos_cu_input_lens, self.item_emb_token_n, self.item_emb_tokens, self.item_llm)
        N = pos_cu_input_lens.size(0)
        pos_embedding = pos_embedding.view(N, -1)

        return pos_embedding
