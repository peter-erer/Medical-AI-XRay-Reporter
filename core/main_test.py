import torch
import argparse
import numpy as np
import os

# 确保导入路径正确
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.base_cmn import BaseCMN  # 你的模型类
from modules.metrics import compute_scores

def test_new_dataset():
    parser = argparse.ArgumentParser()

    # ==========================================
    # 1. 核心路径配置 (请修改这里)
    # ==========================================
    parser.add_argument('--image_dir', type=str, default='你的新图片文件夹路径/', help='新数据的图片文件夹')
    parser.add_argument('--new_ann_path', type=str, default='你的新annotation.json', help='新数据的标注文件')
    
    # 【关键】旧数据的路径 (用于恢复词表)
    parser.add_argument('--old_ann_path', type=str, default='data/iu_xray/annotation.json', help='原版 IU X-Ray 的 json')
    parser.add_argument('--load', type=str, default='results/iu_xray_new_4/model_best.pth', help='模型权重路径')

    # ==========================================
    # 2. 补全缺失的模型参数 (参考你的 main_test.py)
    # ==========================================
    parser.add_argument('--dataset_name', type=str, default='iu_xray')
    parser.add_argument('--max_seq_length', type=int, default=60)
    parser.add_argument('--threshold', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)

    # --- 模型架构参数 ---
    parser.add_argument('--visual_extractor', type=str, default='resnet101')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=2048) # 注意：通常是 2048，如果报错请改回 512
    parser.add_argument('--d_vf', type=int, default=2048) # ResNet 特征维度
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--logit_layers', type=int, default=1)
    parser.add_argument('--bos_idx', type=int, default=0)
    parser.add_argument('--eos_idx', type=int, default=0)
    parser.add_argument('--pad_idx', type=int, default=0)
    parser.add_argument('--use_bn', type=int, default=0)
    
    # 【修复报错的关键参数】
    parser.add_argument('--drop_prob_lm', type=float, default=0.5) 

    # --- CMM 参数 ---
    parser.add_argument('--topk', type=int, default=32)
    parser.add_argument('--cmm_size', type=int, default=2048)
    parser.add_argument('--cmm_dim', type=int, default=512)

    # --- 采样参数 ---
    parser.add_argument('--sample_method', type=str, default='beam_search')
    parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sample_n', type=int, default=1)
    parser.add_argument('--group_size', type=int, default=1)
    parser.add_argument('--output_logsoftmax', type=int, default=1)
    parser.add_argument('--decoding_constraint', type=int, default=0)
    parser.add_argument('--block_trigrams', type=int, default=1)
    
    # --- 其他系统参数 ---
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--seed', type=int, default=7580)
    parser.add_argument('--monitor_mode', type=str, default='max')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4')

    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ==========================================
    # 3. 核心步骤：恢复词表
    # ==========================================
    print(f"1. [Init] 正在从 {args.old_ann_path} 恢复词表...")
    # 临时把 ann_path 换成旧的
    args.ann_path = args.old_ann_path
    
    tokenizer = Tokenizer(args)
    # 【修复】使用 len(tokenizer.idx2token) 避免报错
    print(f"   -> 词表恢复成功，大小: {len(tokenizer.idx2token)} (预期约 761)")
    
    # ==========================================
    # 4. 加载模型
    # ==========================================
    print("2. [Init] 正在初始化模型并加载权重...")
    model = BaseCMN(args, tokenizer)
    model = model.to(device)

    # 加载权重
    checkpoint = torch.load(args.load, map_location=device)
    state_dict = checkpoint['state_dict']
    
    # 处理 DataParallel 的 module. 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.eval()
    print("   -> 模型权重加载完毕！")

    # ==========================================
    # 5. 加载新数据
    # ==========================================
    print(f"3. [Data] 正在加载新数据 {args.new_ann_path} ...")
    # 把 ann_path 指向新文件
    args.ann_path = args.new_ann_path
    
    # 加载 val 集 (因为我们把数据都转到了 val)
    test_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    print(f"   -> 数据加载完成，共 {len(test_dataloader.dataset)} 个样本")

    # ==========================================
    # 6. 推理与可视化
    # ==========================================
    print("4. [Run] 开始推理...")
    res = {}
    gts = {}
    print_count = 0
    
    with torch.no_grad():
        for batch_idx, (ids, images, reports_ids, reports_masks, seq_lengths, retrieved_ids) in enumerate(test_dataloader):
            images = images.to(device)
            reports_ids = reports_ids.to(device)
            
            # 处理检索 ID (新数据无检索库，填空)
            if retrieved_ids is None or retrieved_ids.dim() == 0:
                 retrieved_ids = torch.zeros((images.size(0), args.topk, args.max_seq_length), dtype=torch.long).to(device)
            else:
                 retrieved_ids = retrieved_ids.to(device)

            # 生成
            output, _ = model(images, retrieved_ids, mode='sample')
            
            # 解码
            decoded_res = tokenizer.decode_batch(output.cpu().numpy())
            decoded_gts = tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
            
            for i, (image_id, pred, gt) in enumerate(zip(ids, decoded_res, decoded_gts)):
                res[image_id] = [pred]
                gts[image_id] = [gt]
                
                if print_count < 3:
                    print("-" * 50)
                    print(f"【ID: {image_id}】")
                    print(f"\033[92m[GT]  : {gt}\033[0m")
                    print(f"\033[94m[Pred]: {pred}\033[0m")
                    print_count += 1

    # ==========================================
    # 7. 计算指标
    # ==========================================
    print("\n5. [Eval] 计算指标...")
    scores = compute_scores(gts, res)
    
    print("\n" + "="*30)
    print("  最终测试结果  ")
    print("="*30)
    for metric, score in scores.items():
        print(f"{metric}: {score:.4f}")
    print("="*30)

if __name__ == '__main__':
    test_new_dataset()