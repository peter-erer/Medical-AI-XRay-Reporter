import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# 假设这些是你项目里的 Dataset 类，请确保路径正确
from .datasets import IuxrayMultiImageDataset, MimiccxrSingleImageDataset

# 定义 torchxrayvision 专用的归一化函数
# 目标范围: [-1024, 1024]
def xrv_norm(x):
    return (x - 0.5) * 2048

class R2DataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split

        # === 核心修改区域 ===
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                
                # 随机裁剪 (保持原有逻辑)
                transforms.RandomCrop(224),
                
                # 【关键 1】强制转为单通道 (GrayScale)
                # DenseNet-CheXpert 只能接受单通道输入
                transforms.Grayscale(num_output_channels=1),
                
                # 【关键 2】轻微旋转增强 (3度)
                # 角度很小，避免黑边和几何失真，防止过拟合
                transforms.RandomRotation(degrees=3),
                
                # 转 Tensor (变为 0-1 范围)
                transforms.ToTensor(),
                
                # 【关键 3】使用 XRV 专用数值缩放，代替 ImageNet Normalize
                transforms.Lambda(xrv_norm) 
            ])
        else:
            # 验证集/测试集
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                
                # 测试集也要转单通道
                transforms.Grayscale(num_output_channels=1),
                
                transforms.ToTensor(),
                
                # 测试集也要做同样的数值缩放
                transforms.Lambda(xrv_norm)
            ])
        # === 修改结束 ===

        if self.dataset_name == 'iu_xray':
            self.dataset = IuxrayMultiImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)
        else:
            self.dataset = MimiccxrSingleImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
    # 1. 解包数据 (现在是 6 个元素)
        image_id_batch, image_batch, report_ids_batch, report_masks_batch, seq_lengths_batch, retrieved_ids_batch = zip(*data)

    # 2. 图片堆叠保持不变
        image_batch = torch.stack(image_batch, 0)
    
    # 3. 检索报告堆叠 (Dataset里已Padding过，直接stack)
        retrieved_ids_batch = torch.stack(retrieved_ids_batch, 0)

    # 4. 【关键修正】对目标报告进行动态 Padding
        max_seq_len = max(seq_lengths_batch) # 获取当前 Batch 的最大长度
    
    # 初始化全 0 的数组
        padded_report_ids = np.zeros((len(report_ids_batch), max_seq_len), dtype=np.int64)
        padded_report_masks = np.zeros((len(report_ids_batch), max_seq_len), dtype=np.int64)

        for i, (report_ids, report_masks) in enumerate(zip(report_ids_batch, report_masks_batch)):
            l = len(report_ids)
            padded_report_ids[i, :l] = report_ids
            padded_report_masks[i, :l] = report_masks

    # 5. 转换为 Tensor
        report_ids_batch = torch.from_numpy(padded_report_ids)
        report_masks_batch = torch.from_numpy(padded_report_masks)
        seq_lengths_batch = torch.LongTensor(seq_lengths_batch)

    # 6. 返回 6 个参数，顺序必须与 Trainer 解包顺序一致
        return image_id_batch, image_batch, report_ids_batch, report_masks_batch, seq_lengths_batch, retrieved_ids_batch