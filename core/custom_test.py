import torch
import argparse
import numpy as np
import os
from tqdm import tqdm
import torch.nn.functional as F

try:
    from models.models import BaseCMNModel
except ImportError:
    print("Warning: models.models not found, falling back to modules.base_cmn")
    from modules.base_cmn import BaseCMN as BaseCMNModel

from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores

# ------------------------------------------------------------------
# 1. 辅助类与函数
# ------------------------------------------------------------------

class FlattenVisualExtractorWrapper(torch.nn.Module):
    """
    【最终修复版补丁】
    功能：确保输出永远是 3D [Batch, Seq, Dim]，无论输入是 4D 还是 5D。
    """
    def __init__(self, original_extractor):
        super().__init__()
        self.original = original_extractor

    def forward(self, images):
        # 情况 1: 输入是 5D [B, N, C, H, W] (如果是我们的脚本直接调用)
        if images.dim() == 5:
            b, n, c, h, w = images.size()
            images = images.reshape(b * n, c, h, w)
            att_feats, fc_feats = self.original(images)
            
            # CNN 输出通常是 [B*N, C, H, W] (4D)
            if att_feats.dim() == 4:
                # 拍平: [B*N, C, H, W] -> [B*N, H*W, C]
                b_n, c, h_out, w_out = att_feats.size()
                att_feats = att_feats.permute(0, 2, 3, 1) 
                att_feats = att_feats.reshape(b_n, h_out * w_out, c)
            
            # 恢复 Views 维度: [B, N, Seq, Dim]
            att_feats = att_feats.reshape(b, n, -1, att_feats.size(-1))
            # 合并 Views 和 Seq: [B, N*Seq, Dim] -> 最终 3D
            att_feats = att_feats.reshape(b, -1, att_feats.size(-1))

        # 情况 2: 输入是 4D [B, C, H, W] (models.py 拆分后调用) -> 报错就在这里修复
        else:
            att_feats, fc_feats = self.original(images)
            
            # 关键修复：如果输出是 4D [B, C, H, W]，必须拍平为 3D [B, H*W, C]
            if att_feats.dim() == 4:
                b, c, h_out, w_out = att_feats.size()
                # 1. 把 Channel 放到最后: [B, H, W, C]
                att_feats = att_feats.permute(0, 2, 3, 1) 
                # 2. 拍平 H 和 W: [B, H*W, C]
                att_feats = att_feats.reshape(b, h_out * w_out, c)
            
        return att_feats, fc_feats

def extract_features_mean(model, images):
    """
    提取特征用于检索
    """
    # images shape: [Batch, Views, C, H, W]
    b = images.size(0) 
    
    # 这里的 model.visual_extractor 已经是 Wrapper 了
    # 它能接受 5D 输入，并返回拍平的 att_feats 和原始的 fc_feats ([B*N, 2048])
    _, fc_feats = model.visual_extractor(images)
    
    # 这里的 fc_feats 是 [B*N, 2048]，我们需要把它变回 [B, N, 2048]
    # 使用 reshape 防止报错
    fc_feats = fc_feats.reshape(b, -1, 2048) 
    
    # 对多张图取平均
    fc_feats_mean = fc_feats.mean(dim=1)  # [B, 2048]
    fc_feats_mean = F.normalize(fc_feats_mean, dim=-1)
    
    return fc_feats_mean

def build_memory_bank(model, data_loader, device, args):
    print("正在构建检索记忆库 (Memory Bank)...")
    model.eval()
    
    memory_bank_feats = []
    memory_bank_reports = []
    
    with torch.no_grad():
        for i, (ids, images, reports_ids, _, _, _) in enumerate(tqdm(data_loader, desc="Encoding Train Set")):
            images = images.to(device)
            
            # 1. 提取视觉特征
            fc_feats = extract_features_mean(model, images)
            
            # 2. 统一报告长度
            batch_size, current_len = reports_ids.shape
            target_len = args.max_seq_length
            
            if current_len < target_len:
                pad_size = target_len - current_len
                
                # 【修复】让 padding 的设备与 reports_ids 保持一致 (CPU)
                padding = torch.full(
                    (batch_size, pad_size), 
                    args.pad_idx, 
                    dtype=reports_ids.dtype,
                    device=reports_ids.device  # <--- 关键修改：跟随 reports_ids
                )
                
                reports_ids = torch.cat([reports_ids, padding], dim=1)
                
            elif current_len > target_len:
                reports_ids = reports_ids[:, :target_len]
            
            memory_bank_feats.append(fc_feats.cpu())
            memory_bank_reports.append(reports_ids.cpu())
            
    memory_bank_feats = torch.cat(memory_bank_feats, dim=0).to(device)
    memory_bank_reports = torch.cat(memory_bank_reports, dim=0).to(device)
    
    print(f"记忆库构建完成！包含 {memory_bank_feats.size(0)} 个历史病例。")
    return memory_bank_feats, memory_bank_reports

def perform_retrieval(current_feats, memory_feats, memory_reports, topk=32):
    scores = torch.matmul(current_feats, memory_feats.t())
    _, topk_indices = torch.topk(scores, k=topk, dim=-1)
    batch_size, k = topk_indices.size()
    seq_len = memory_reports.size(1)
    flat_indices = topk_indices.view(-1)
    retrieved_reports = memory_reports.index_select(0, flat_indices)
    retrieved_reports = retrieved_reports.view(batch_size, k, seq_len)
    return retrieved_reports

# ------------------------------------------------------------------
# 2. 主函数
# ------------------------------------------------------------------
def test_new_dataset_with_retrieval():
    parser = argparse.ArgumentParser()
    # ... (前面的参数定义部分保持不变，省略以节省空间) ...
    # 确保你的参数定义都在，不需要改参数，只需要改下面的 Loop 逻辑
    # ...
    parser.add_argument('--image_dir', type=str, default='你的新图片文件夹路径/')
    parser.add_argument('--new_ann_path', type=str, default='你的新annotation.json')
    parser.add_argument('--train_image_dir', type=str, default='data/iu_xray/images/')
    parser.add_argument('--old_ann_path', type=str, default='data/iu_xray/annotation.json')
    parser.add_argument('--load', type=str, default='results/iu_xray_new_4/model_best.pth')
    
    parser.add_argument('--dataset_name', type=str, default='iu_xray')
    parser.add_argument('--max_seq_length', type=int, default=60)
    parser.add_argument('--threshold', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--visual_extractor', type=str, default='resnet101')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--d_vf', type=int, default=2048) 
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--logit_layers', type=int, default=1)
    parser.add_argument('--bos_idx', type=int, default=0)
    parser.add_argument('--eos_idx', type=int, default=0)
    parser.add_argument('--pad_idx', type=int, default=0)
    parser.add_argument('--use_bn', type=int, default=0)
    parser.add_argument('--drop_prob_lm', type=float, default=0.5) 
    parser.add_argument('--topk', type=int, default=32)
    parser.add_argument('--cmm_size', type=int, default=2048)
    parser.add_argument('--cmm_dim', type=int, default=512)
    parser.add_argument('--sample_method', type=str, default='beam_search')
    parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sample_n', type=int, default=1)
    parser.add_argument('--group_size', type=int, default=1)
    parser.add_argument('--output_logsoftmax', type=int, default=1)
    parser.add_argument('--decoding_constraint', type=int, default=0)
    parser.add_argument('--block_trigrams', type=int, default=1)
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--seed', type=int, default=7580)
    parser.add_argument('--monitor_mode', type=str, default='max')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"1. [Init] 恢复词表...")
    args.ann_path = args.old_ann_path
    tokenizer = Tokenizer(args)
    print(f"   -> 词表大小: {len(tokenizer.idx2token)}")
    
    print("2. [Init] 初始化模型...")
    model = BaseCMNModel(args, tokenizer)
    model = model.to(device)

    checkpoint = torch.load(args.load, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(new_state_dict, strict=False)
    
    print("   -> 应用 VisualExtractor 维度修复补丁...")
    model.visual_extractor = FlattenVisualExtractorWrapper(model.visual_extractor)
    
    model.eval()
    print("   -> 模型就绪！")

    print("3. [Retrieval] 构建记忆库...")
    temp_image_dir = args.image_dir
    args.image_dir = args.train_image_dir
    args.ann_path = args.old_ann_path
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=False)
    
    memory_feats, memory_reports = build_memory_bank(model, train_dataloader, device, args)
    
    args.image_dir = temp_image_dir 

    print(f"4. [Data] 加载新数据...")
    args.ann_path = args.new_ann_path
    test_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    
    print("5. [Run] 开始推理...")
    res, gts = {}, {}
    print_count = 0
    
    with torch.no_grad():
        for batch_idx, (ids, images, reports_ids, _, _, _) in enumerate(test_dataloader):
            images = images.to(device)
            reports_ids = reports_ids.to(device)
            
            current_feats = extract_features_mean(model, images)
            retrieved_ids = perform_retrieval(current_feats, memory_feats, memory_reports, topk=args.topk)
            
            # 【关键修改在这里】
            retrieved_ids = retrieved_ids.view(retrieved_ids.size(0), -1)

            # ... 在循环里 ...
            
            # 定义参数字典
            sample_opts = {
                'sample_method': 'sample', # 强制改为 sample
                'beam_size': 1,
                'temperature': 1.0,        # 尝试 0.8 ~ 1.2
                'sample_n': 1
            }

            # 传入参数
            output, _ = model(images, retrieved_ids, mode='sample', update_opts=sample_opts)
            
            decoded_res = tokenizer.decode_batch(output.cpu().numpy())
            decoded_gts = tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
            
            for i, (image_id, pred, gt) in enumerate(zip(ids, decoded_res, decoded_gts)):
                res[image_id] = [pred]
                gts[image_id] = [gt]
                if print_count < 5:
                    print("-" * 50)
                    print(f"【ID: {image_id}】")
                    print(f"\033[92m[GT]  : {gt}\033[0m")
                    print(f"\033[94m[Pred]: {pred}\033[0m")
                    print_count += 1

    print("\n6. [Eval] 计算指标...")
    scores = compute_scores(gts, res)
    for metric, score in scores.items():
        print(f"{metric}: {score:.4f}")

if __name__ == '__main__':
    test_new_dataset_with_retrieval()