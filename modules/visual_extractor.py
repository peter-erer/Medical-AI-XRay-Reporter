import torch
import torch.nn as nn
import torchxrayvision as xrv

class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        # 这里的 args 参数虽然传入，但我们为了效果强制使用 DenseNet121-CheXpert
        self.visual_extractor = "densenet121"
        self.pretrained = True
        
        print("=== 初始化 Visual Extractor ===")
        print("正在加载 torchxrayvision (CheXpert) DenseNet121 权重...")
        
        # 1. 加载 torchxrayvision 模型
        # weights="densenet121-res224-chex": 指定 CheXpert 数据集预训练
        xrv_model = xrv.models.DenseNet(weights="densenet121-res224-chex")
        
        # 2. 提取卷积特征部分 (丢弃分类头)
        self.model = xrv_model.features
        
        # 3. 【关键适配层】维度转换
        # DenseNet121 输出通道是 1024
        # R2GEN-CMN (Transformer) 通常期望输入是 2048 (ResNet的标准)
        # 使用 1x1 卷积将 1024 升维到 2048，避免修改后续 Transformer 代码
        self.feature_adaptor = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=1),
            nn.BatchNorm2d(2048),
            nn.ReLU()
        )
        
        # 4. 平均池化层 (保持原样)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        
        print("Visual Extractor 加载完成。特征维度已适配: 1024 -> 2048")

    def forward(self, images):
        # images shape: [Batch, 1, 224, 224] (注意：必须是单通道)
        
        # 1. 提取基础特征
        patch_feats = self.model(images) 
        # Output: [Batch, 1024, 7, 7]
        
        # 2. 维度适配 (1024 -> 2048)
        patch_feats = self.feature_adaptor(patch_feats)
        # Output: [Batch, 2048, 7, 7]
        
        # 3. 计算全局特征 (Global Average Pooling)
        # Output: [Batch, 2048]
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        
        # 4. 调整 Patch 特征格式供 Transformer 使用
        batch_size, feat_size, _, _ = patch_feats.shape
        # Flatten spatial dims: [Batch, 2048, 49] -> [Batch, 49, 2048]
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        
        return patch_feats, avg_feats