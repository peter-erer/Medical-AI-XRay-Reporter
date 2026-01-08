import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import resnet101, ResNet101_Weights
from PIL import Image
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import NearestNeighbors

# 配置路径 - 指向包含大量 CXRxxx 文件夹的根目录
IMG_ROOT = './data/iu_xray/images/' 
SAVE_PATH = './retrieval_index.json'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_extraction():
    # 1. 加载模型 (使用最新 Weights 语法)
    weights = ResNet101_Weights.DEFAULT
    resnet = resnet101(weights=weights)
    model = nn.Sequential(*list(resnet.children())[:-1]).to(DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    study_ids = []
    all_study_features = []

    # 获取所有子文件夹 (CXRxxx)
    subfolders = [d for d in os.listdir(IMG_ROOT) if os.path.isdir(os.path.join(IMG_ROOT, d))]
    print(f"找到 {len(subfolders)} 个病例目录。")

    print("--- 正在按病例提取特征 ---")
    with torch.no_grad():
        for folder_name in tqdm(subfolders):
            folder_path = os.path.join(IMG_ROOT, folder_name)
            img_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not img_files:
                continue

            current_study_feats = []
            for img_name in img_files:
                img_path = os.path.join(folder_path, img_name)
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(DEVICE)
                feat = model(img_tensor).cpu().numpy().flatten()
                current_study_feats.append(feat)
            
            # 将该病例下的所有图片特征取平均，作为该 Study 的全局特征
            study_feat = np.mean(current_study_feats, axis=0)
            
            all_study_features.append(study_feat)
            study_ids.append(folder_name)

    # 2. KNN 计算 (针对 Study 特征)
    print("--- 正在构建索引 ---")
    all_study_features = np.array(all_study_features)
    knn = NearestNeighbors(n_neighbors=2, metric='cosine') 
    knn.fit(all_study_features)
    _, indices = knn.kneighbors(all_study_features)

    # 3. 保存结果
    # 结果格式：{"CXR123": ["CXR456"]} 表示与 CXR123 最像的病例是 CXR456
    res = {study_ids[i]: [study_ids[idx] for idx in indices[i][1:]] for i in range(len(study_ids))}
    
    with open(SAVE_PATH, 'w') as f:
        json.dump(res, f)
    
    print(f"成功！索引已保存至 {SAVE_PATH}，共处理 {len(study_ids)} 个病例。")

if __name__ == '__main__':
    run_extraction()