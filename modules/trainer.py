import logging
import os
from abc import abstractmethod
import torch
import numpy as np
from numpy import inf
from modules.loss import compute_scst_loss # 确保你在 loss.py 中定义了此函数
from tqdm import tqdm

class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler):
        self.args = args
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period
        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)
        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if args.resume is not None:
            self._resume_checkpoint(args.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)

            for key, value in log.items():
                self.logger.info('\t{:15s}: {}'.format(str(key), value))

            best = False
            if self.mnt_mode != 'off':
                try:
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Training stops due to early stopping.")
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val: self.best_recorder['val'].update(log)
        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][self.mnt_metric_test])
        if improved_test: self.best_recorder['test'].update(log)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        device = torch.device('cuda:0' if n_gpu_use > 0 and n_gpu > 0 else 'cpu')
        return device, list(range(n_gpu_use))

    def _save_checkpoint(self, epoch, save_best=False):
        state = {'epoch': epoch, 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'monitor_best': self.mnt_best}
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        if save_best:
            torch.save(state, os.path.join(self.checkpoint_dir, 'model_best.pth'))

    def _resume_checkpoint(self, resume_path):
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args, lr_scheduler)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

# 在 modules/trainer.py 中替换此方法
    def _get_self_critical_reward(self, sample_reports, greedy_reports, ground_truths):
        import numpy as np
        
        # 定义一个简单的 Dice/F1 计算函数
        def compute_score(pred, gt):
            pred_toks = set(pred.split())
            gt_toks = set(gt.split())
            if len(pred_toks) == 0: return 0.0
            
            # 计算重合词数
            intersection = len(pred_toks & gt_toks)
            # Dice Coefficient formula: 2 * (A ∩ B) / (|A| + |B|)
            return 2.0 * intersection / (len(pred_toks) + len(gt_toks) + 1e-6)
        
        rewards = []
        for i in range(len(ground_truths)):
            gt = ground_truths[i] # 这是一个长字符串
            sample = sample_reports[i]
            greedy = greedy_reports[i]
            
            s_score = compute_score(sample, gt)
            g_score = compute_score(greedy, gt)
            
            rewards.append(s_score - g_score)
            
        return np.array(rewards)

    def _train_epoch(self, epoch):
        # 判断是否进入强化学习阶段 (通常在预训练若干轮后开启)
        if getattr(self.args, 'use_scst', False) and epoch >= getattr(self.args, 'scst_start_epoch', 0):
            return self._train_epoch_rl(epoch)
        
        # 否则执行标准的交叉熵训练 (你原有的逻辑)
        self.logger.info('[{}/{}] Training with Cross Entropy.'.format(epoch, self.epochs))
        train_loss = 0
        self.model.train()
        for batch_idx, (ids, images, reports_ids, reports_masks, seq_lengths, retrieved_ids) in enumerate(self.train_dataloader):
            images, reports_ids, reports_masks, retrieved_ids = images.to(self.device), reports_ids.to(self.device), \
                                                               reports_masks.to(self.device), retrieved_ids.to(self.device)
            output = self.model(images, retrieved_ids, reports_ids, mode='train')
            loss = self.criterion(output, reports_ids, reports_masks)
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        log = {'train_loss': train_loss / len(self.train_dataloader)}
        log.update(self._evaluation(epoch))
        self.lr_scheduler.step()
        return log

    def _train_epoch_rl(self, epoch):
        """SCST 强化学习训练循环 (带进度条版)"""
        self.logger.info('[{}/{}] Training with SCST (RL).'.format(epoch, self.epochs))
        self.model.train()
        total_reward = 0
        
        # 使用 tqdm 包装 dataloader，显示进度条
        # desc: 进度条左边的文字
        # leave: 跑完是否保留进度条
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch} SCST", leave=True)

        for batch_idx, (ids, images, reports_ids, _, _, retrieved_ids) in enumerate(pbar):
            images, retrieved_ids = images.to(self.device), retrieved_ids.to(self.device)

            # 1. Greedy 搜索 (Baseline)
            self.model.eval()
            with torch.no_grad():
                greedy_res_ids, _ = self.model(images, retrieved_ids, mode='sample', sample_method='greedy')
            self.model.train()

            # 2. Sample 采样 (Exploration)
            sample_res_ids, sample_log_probs = self.model(images, retrieved_ids, mode='sample', sample_method='sample')

            # 3. 计算 Reward (这一步可能是性能瓶颈！)
            greedy_reports = self.model.tokenizer.decode_batch(greedy_res_ids.cpu().numpy())
            sample_reports = self.model.tokenizer.decode_batch(sample_res_ids.cpu().numpy())
            ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
            if batch_idx == 0:
                print("\n" + "="*50)
                print(f"[Epoch {epoch} Diagnosis]")
                print(f"GT    : {ground_truths[0]}")
                print(f"Greedy: {greedy_reports[0]}")
                print(f"Sample: {sample_reports[0]}")
                print("-" * 20)
                print(f"GT    : {ground_truths[1]}")
                print(f"Greedy: {greedy_reports[1]}")
                print(f"Sample: {sample_reports[1]}")
                print("="*50 + "\n")
            
            # ... 继续计算 Reward ...
            reward = self._get_self_critical_reward(sample_reports, greedy_reports, ground_truths)
            reward = torch.from_numpy(reward).float().to(self.device)
            if reward.std() > 0:
                reward = (reward - reward.mean()) / (reward.std() + 1e-6)
            
            # 记录当前 batch 的平均奖励
            current_reward = reward.mean().item()
            total_reward += current_reward

            # 4. 计算 Loss 并更新
            from modules.loss import compute_scst_loss
            loss = compute_scst_loss(sample_log_probs, sample_res_ids, reward)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 【可视化核心】实时更新进度条右侧信息
            # Avg R: 当前累计的平均奖励
            # Cur R: 当前 Batch 的奖励 (观察这个值是否在变)
            pbar.set_postfix({'Avg R': total_reward / (batch_idx + 1), 'Cur R': current_reward})

        log = {'train_reward': total_reward / len(self.train_dataloader)}
        log.update(self._evaluation(epoch))
        self.lr_scheduler.step()
        return log

    def _evaluation(self, epoch):
        """评估逻辑 (Val & Test)"""
        self.model.eval()
        val_log = {}
        with torch.no_grad():
            for split in ['val', 'test']:
                dataloader = getattr(self, f'{split}_dataloader')
                gts, res = [], []
                for batch_idx, (ids, images, reports_ids, reports_masks, _, retrieved_ids) in enumerate(dataloader):
                    images, retrieved_ids = images.to(self.device), retrieved_ids.to(self.device)
                    output, _ = self.model(images, retrieved_ids, mode='sample')
                    res.extend(self.model.tokenizer.decode_batch(output.cpu().numpy()))
                    gts.extend(self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy()))
                
                met = self.metric_ftns({i: [gt] for i, gt in enumerate(gts)}, {i: [r] for i, r in enumerate(res)})
                val_log.update({f'{split}_' + k: v for k, v in met.items()})
        return val_log