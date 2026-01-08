import json
import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        
        # 加载原始标注
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.examples = self.ann[self.split]
        
        # --- 新增：加载检索索引 ---
        # 假设你把 retrieval_index.json 放在了项目根目录
        index_path = './retrieval_index.json'
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                self.retrieval_index = json.load(f)
        else:
            self.retrieval_index = {}
            print(f"Warning: {index_path} not found!")

        # --- 新增：建立 Study ID 到报告的全局映射 ---
        # 将 train, val, test 所有数据的 ID 和报告存入字典，方便跨分片检索
        self.id2report = {}
        for s in ['train', 'val', 'test']:
            for item in self.ann[s]:
                self.id2report[item['id']] = item['report']

        # 预处理当前 split 的目标报告
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)

    # 辅助函数：处理检索文本的 Tokenize 和 Padding
    def get_retrieved_ids(self, current_id):
        # 1. 查找邻居 ID，找不到就回退到自己
        neighbor_ids = self.retrieval_index.get(current_id, [current_id])
        neighbor_id = neighbor_ids[0]
        
        # 2. 获取报告文本并 Tokenize
        neighbor_report = self.id2report.get(neighbor_id, "")
        retrieved_ids = self.tokenizer(neighbor_report)[:self.max_seq_length]
        
        # 3. 必须进行 Padding，否则 DataLoader 在拼 Batch 时会因为长度不一报错
        padding_len = self.max_seq_length - len(retrieved_ids)
        retrieved_ids = retrieved_ids + [0] * padding_len
        
        return torch.tensor(retrieved_ids, dtype=torch.long)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        
        # 加载双图
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if len(image_path) > 1:
            image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        else:
            # 如果只有一张图，直接复用第一张图当作第二张图
            image_2 = image_1.copy()
        
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        
        image = torch.stack((image_1, image_2), 0)
        
        # 目标报告
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        
        # --- 新增：获取检索报告的 Tensor ---
        retrieved_ids = self.get_retrieved_ids(image_id)
        
        # 返回 sample，增加了第 6 个元素 retrieved_ids
        sample = (image_id, image, report_ids, report_masks, seq_length, retrieved_ids)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        # 原代码这行 image_id 赋值似乎覆盖了原始 ID，保持原样或根据需要修改
        # image_id = os.path.join(self.image_dir, image_path[0]) 
        
        if self.transform is not None:
            image = self.transform(image)
            
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        
        # --- 新增：获取检索报告 ---
        retrieved_ids = self.get_retrieved_ids(example['id'])
        
        sample = (image_id, image, report_ids, report_masks, seq_length, retrieved_ids)
        return sample