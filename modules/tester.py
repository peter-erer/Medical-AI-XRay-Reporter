import logging
import os
from abc import abstractmethod

import cv2
import numpy as np
import pandas as pd
import spacy
import torch
from tqdm import tqdm

from modules.utils import generate_heatmap


class BaseTester(object):
    def __init__(self, model, criterion, metric_ftns, args):
        self.args = args

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns

        self.epochs = self.args.epochs
        self.save_dir = self.args.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self._load_checkpoint(args.load)

    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def plot(self):
        raise NotImplementedError

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _load_checkpoint(self, load_path):
        load_path = str(load_path)
        self.logger.info("Loading checkpoint: {} ...".format(load_path))
        checkpoint = torch.load(load_path)
        # 如果模型包含在 DataParallel 中，加载时可能需要处理前缀
        self.model.load_state_dict(checkpoint['state_dict'])


class Tester(BaseTester):
    def __init__(self, model, criterion, metric_ftns, args, test_dataloader):
        super(Tester, self).__init__(model, criterion, metric_ftns, args)
        self.test_dataloader = test_dataloader

    def test(self):
        self.logger.info('Start to evaluate in the test set.')
        self.model.eval()
        log = dict()
        
        # =========================================================
        # 【关键修复】构建采样参数字典，强行覆盖模型默认参数
        # =========================================================
        sample_opts = {
            'sample_method': self.args.sample_method,
            'beam_size': self.args.beam_size,
            'temperature': self.args.temperature,
            'sample_n': self.args.sample_n,
            'group_size': self.args.group_size,
            'decoding_constraint': self.args.decoding_constraint,
            'block_trigrams': self.args.block_trigrams
        }
        self.logger.info(f"Using sampling options: {sample_opts}")
        # =========================================================

        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks, seq_lengths, retrieved_ids) in tqdm(enumerate(self.test_dataloader)):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                retrieved_ids = retrieved_ids.to(self.device)

                # 【关键修复】将 sample_opts 传入 update_opts 参数
                # 注意：AttModel 的 forward 函数会将 kwargs 传给 _sample
                # 我们需要查看 models.py (你之前没发这个) 的 forward 怎么写的
                # 但通常 CaptionModel 的 forward(..., mode='sample', update_opts={...}) 是标准写法
                
                output, _ = self.model(images, retrieved_ids, mode='sample', update_opts=sample_opts)
                
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)

            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            print(log)

            test_res_df, test_gts_df = pd.DataFrame(test_res), pd.DataFrame(test_gts)
            test_res_df.to_csv(os.path.join(self.save_dir, "res.csv"), index=False, header=False)
            test_gts_df.to_csv(os.path.join(self.save_dir, "gts.csv"), index=False, header=False)

        return log

    def plot(self):
        assert self.args.batch_size == 1 and self.args.beam_size == 1
        self.logger.info('Start to plot attention weights in the test set.')
        os.makedirs(os.path.join(self.save_dir, "attentions"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "attentions_entities"), exist_ok=True)
        # 确保已安装 spacy 科学模型：python -m spacy download en_core_sci_sm
        ner = spacy.load("en_core_sci_sm")
        mean = torch.tensor((0.485, 0.456, 0.406))
        std = torch.tensor((0.229, 0.224, 0.225))
        mean = mean[:, None, None]
        std = std[:, None, None]

        self.model.eval()
        with torch.no_grad():
            # 修改：解包 6 个参数
            for batch_idx, (images_id, images, reports_ids, reports_masks, seq_lengths, retrieved_ids) in tqdm(enumerate(self.test_dataloader)):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                retrieved_ids = retrieved_ids.to(self.device)

                # 修改：模型调用
                output, _ = self.model(images, retrieved_ids, mode='sample')
                
                image = torch.clamp((images[0].cpu() * std + mean) * 255, 0, 255).int().cpu().numpy()
                report = self.model.tokenizer.decode_batch(output.cpu().numpy())[0].split()

                char2word = [idx for word_idx, word in enumerate(report) for idx in [word_idx] * (len(word) + 1)][:-1]

                # 注意：这里需要 BaseCMN 的 _save_attns 逻辑正常执行
                attention_weights = self.model.encoder_decoder.attention_weights[:-1]
                
                # 检查逻辑：由于 RAG 在 att_feats 开头拼了一个词向量，
                # 这里的注意力权重空间维度会多出 1。可视化时我们只取后段的视觉部分。
                
                assert len(attention_weights) == len(report)
                for word_idx, (attns, word) in enumerate(zip(attention_weights, report)):
                    for layer_idx, attn in enumerate(attns):
                        # attn 形状假设为 [1, num_heads, seq_len]
                        # seq_len 现在是 (视觉Token数 + 1)
                        # 我们跳过第一个 Token (检索文本)，只可视化视觉相关的注意力
                        visual_attn = attn[:, :, 1:] 
                        
                        os.makedirs(os.path.join(self.save_dir, "attentions", "{:04d}".format(batch_idx),
                                                 "layer_{}".format(layer_idx)), exist_ok=True)

                        heatmap = generate_heatmap(image, visual_attn.mean(1).squeeze())
                        cv2.imwrite(os.path.join(self.save_dir, "attentions", "{:04d}".format(batch_idx),
                                                 "layer_{}".format(layer_idx), "{:04d}_{}.png".format(word_idx, word)),
                                    heatmap)

                # 实体级别的注意力可视化
                for ne_idx, ne in enumerate(ner(" ".join(report)).ents):
                    for layer_idx in range(len(attention_weights[0])):
                        os.makedirs(os.path.join(self.save_dir, "attentions_entities", "{:04d}".format(batch_idx),
                                                 "layer_{}".format(layer_idx)), exist_ok=True)
                        
                        # 同样的切片逻辑：跳过第一个 Token [:, :, 1:]
                        attn_slice = [attns[layer_idx][:, :, 1:] for attns in
                                     attention_weights[char2word[ne.start_char]:char2word[ne.end_char] + 1]]
                        attn_concat = np.concatenate(attn_slice, axis=2)
                        
                        heatmap = generate_heatmap(image, attn_concat.mean(1).mean(1).squeeze())
                        cv2.imwrite(os.path.join(self.save_dir, "attentions_entities", "{:04d}".format(batch_idx),
                                                 "layer_{}".format(layer_idx), "{:04d}_{}.png".format(ne_idx, ne)),
                                    heatmap)
        return {}