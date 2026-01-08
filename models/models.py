import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.base_cmn import BaseCMN
from modules.visual_extractor import VisualExtractor

class BaseCMNModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(BaseCMNModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = BaseCMN(args, tokenizer)
        
        # 1. 检索特征升维层 (512 -> 2048)
        # 必须保留这个，因为 checkpoint 里有这一层的权重
        self.retrieval_up_project = nn.Linear(args.d_model, args.d_vf)
        
        # 2. FC 降维层 (IU Xray 双图 4096 -> 2048)
        if args.dataset_name == 'iu_xray':
            self.fc_reduce = nn.Linear(args.d_vf * 2, args.d_vf)

        # 【注意】这里去掉了 gate_linear，以匹配你的 checkpoint

        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, retrieved_ids, targets=None, mode='train', update_opts={}, **kwargs):
        # 兼容性处理：如果 trainer 传了 sample_method，可以在 kwargs 里接收
        
        # 1. 提取视觉特征
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1) # [B, 4096]
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        
        # 降维 fc_feats 到 2048
        fc_feats = self.fc_reduce(fc_feats)

        # 2. 处理检索报告特征
        embed_layer = self.encoder_decoder.model.tgt_embed[0].lut
        retrieved_feats = embed_layer(retrieved_ids) 
        
        # 升维: [B, 1, 512] -> [B, 1, 2048]
        retrieved_global = torch.mean(retrieved_feats, dim=1, keepdim=True)
        retrieved_global = self.retrieval_up_project(retrieved_global)

        # 3. 直接拼接融合 (无 Gate)
        att_feats = torch.cat((retrieved_global, att_feats), dim=1)

        # 4. 后续逻辑
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return output
        elif mode == 'sample':
            # 将 kwargs 里的 sample_method 传给 encoder_decoder._sample
            # BaseCMN 的 forward 会根据 mode='sample' 调用 _sample
            output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample', update_opts=update_opts, **kwargs)
            return output, output_probs
        else:
            raise ValueError

    def forward_mimic_cxr(self, images, retrieved_ids, targets=None, mode='train', update_opts={}, **kwargs):
        att_feats, fc_feats = self.visual_extractor(images)

        # 检索处理
        embed_layer = self.encoder_decoder.model.tgt_embed[0].lut
        retrieved_feats = embed_layer(retrieved_ids)
        retrieved_global = torch.mean(retrieved_feats, dim=1, keepdim=True)
        retrieved_global = self.retrieval_up_project(retrieved_global)

        # 直接拼接
        att_feats = torch.cat((retrieved_global, att_feats), dim=1)

        if mode == 'train':
            return self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            return self.encoder_decoder(fc_feats, att_feats, mode='sample', update_opts=update_opts, **kwargs)
        else:
            raise ValueError