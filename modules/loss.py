import torch
import torch.nn as nn

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # --- 交叉熵训练 (CE) 用 ---
        # input: [batch, seq_len, vocab_size] (3维)
        # target: [batch, seq_len]
        
        # 这里的 input 是完整的 logits，所以需要 gather 提取对应目标词的概率
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        # --- 强化学习训练 (RL) 用 ---
        # input: [batch, seq_len] (2维) -> 这里已经是采样得到的 log_probs 了
        # seq: [batch, seq_len]
        # reward: [batch]
        
        # 【关键修改】不要 gather，直接计算
        # input 已经是 log(P(selected_word))，直接乘以 reward 即可
        mask = (seq > 0).float()
        
        # Loss = -log_prob * reward
        # reward 需要 unsqueeze 变成 [batch, 1] 以便广播
        output = -input * reward.unsqueeze(1) * mask
        
        output = torch.sum(output) / torch.sum(mask)
        return output

def compute_loss(output, reports_ids, reports_masks):
    # CE 阶段调用
    criterion = LanguageModelCriterion()
    loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
    return loss

def compute_scst_loss(input, seq, reward):
    # RL 阶段调用
    criterion = RewardCriterion()
    return criterion(input, seq, reward)