# -*- coding: utf-8 -*-
import os
import random
import csv
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.models as models

from transformers import BertTokenizer, BertModel

SEED = 42
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 3
MAX_TEXT_LENGTH = 128

def set_seed(seed=42):
    """
    设置随机种子，保证可复现性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

def read_data_from_txt(file_path):
    """
    读取指定文件并返回 (guid, 标签) 的列表。
    """
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            guid, tag = row
            data_list.append((guid, tag))
    return data_list

def safe_read_text(text_path):
    """
    先尝试以 UTF-8 读取文本，若解码出错则转用 GB18030 并忽略非法字符。
    """
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except UnicodeDecodeError:
        with open(text_path, 'r', encoding='gb18030', errors='ignore') as f:
            return f.read().strip()

class MultiModalDataset(Dataset):
    """
    多模态数据集：读取 guid 对应的文本和图像，以及标签。
    """
    def __init__(self, data_list, data_dir, tokenizer, max_length=128, is_test=False):
        """
        初始化多模态数据集，包括文本 tokenizer、图像变换以及标签映射。
        """
        super().__init__()
        self.data_list = data_list
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        self.label2id = {"negative": 0, "neutral": 1, "positive": 2}
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.img_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        """
        返回数据集大小。
        """
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        返回对应 idx 的文本张量、图像张量与标签。
        """
        guid, tag = self.data_list[idx]
        text_path = os.path.join(self.data_dir, f"{guid}.txt")
        text_str = safe_read_text(text_path)
        encoding = self.tokenizer(
            text_str,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        img_path = os.path.join(self.data_dir, f"{guid}.jpg")
        image = Image.open(img_path).convert('RGB')
        image = self.img_transform(image)
        if not self.is_test:
            label_id = self.label2id[tag]
        else:
            label_id = -1
        return {
            "guid": guid,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image,
            "label": label_id
        }

class MultiModalModel(nn.Module):
    """
    多模态模型：文本用 BERT，图像用 ResNet，拼接后分类。
    """
    def __init__(self, num_classes=3, text_model_name="bert-base-chinese",
                 use_text=True, use_image=True):
        """
        初始化多模态模型，包括文本编码器、图像编码器以及融合后的分类层。
        """
        super().__init__()
        self.use_text = use_text
        self.use_image = use_image
        if self.use_text:
            self.text_encoder = BertModel.from_pretrained(text_model_name)
            text_hidden_size = self.text_encoder.config.hidden_size
        else:
            text_hidden_size = 0
        if self.use_image:
            self.img_encoder = models.resnet18(pretrained=True)
            self.img_encoder.fc = nn.Identity()
            img_hidden_size = 512
        else:
            img_hidden_size = 0
        fusion_dim = text_hidden_size + img_hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask, images):
        """
        前向传播，返回分类 logits。
        """
        if self.use_text:
            text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_feat = text_outputs.pooler_output
        else:
            text_feat = torch.zeros(input_ids.size(0), 0, device=input_ids.device)
        if self.use_image:
            img_feat = self.img_encoder(images)
        else:
            img_feat = torch.zeros(images.size(0), 0, device=images.device)
        fusion_feat = torch.cat([text_feat, img_feat], dim=1)
        logits = self.classifier(fusion_feat)
        return logits

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    训练单个 Epoch 并返回平均损失与准确率。
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        _, preds = torch.max(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

def validate_model(model, dataloader, criterion, device):
    """
    在验证集上评估模型，返回平均损失与准确率。
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            logits = model(input_ids, attention_mask, images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            _, preds = torch.max(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

def predict_test(model, dataloader, device):
    """
    对测试集进行推理，并返回 (guid, 预测情感标签) 列表。
    """
    model.eval()
    predictions = []
    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    with torch.no_grad():
        for batch in dataloader:
            guids = batch['guid']
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            logits = model(input_ids, attention_mask, images)
            _, preds = torch.max(logits, dim=1)
            for guid, pred_id in zip(guids, preds):
                predictions.append((guid, id2label[pred_id.item()]))
    return predictions

def main():
    """
    主函数：解析命令行参数并执行训练、验证和可选的测试推理流程。
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="train.txt")
    parser.add_argument("--test_file", type=str, default="test_without_label.txt")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--text_model_name", type=str, default="bert-base-chinese")
    parser.add_argument("--use_text", action='store_true', default=True)
    parser.add_argument("--use_image", action='store_true', default=True)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--skip_test", action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_list = read_data_from_txt(args.train_file)
    random.shuffle(train_list)
    train_size = int(0.8 * len(train_list))
    val_size = len(train_list) - train_size
    train_data = train_list[:train_size]
    val_data = train_list[train_size:]
    tokenizer = BertTokenizer.from_pretrained(args.text_model_name)
    train_dataset = MultiModalDataset(train_data, args.data_dir, tokenizer, max_length=MAX_TEXT_LENGTH, is_test=False)
    val_dataset = MultiModalDataset(val_data, args.data_dir, tokenizer, max_length=MAX_TEXT_LENGTH, is_test=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = MultiModalModel(
        num_classes=3,
        text_model_name=args.text_model_name,
        use_text=args.use_text,
        use_image=args.use_image
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")
            print(">> Model saved.")

    print("Training finished!")
    print(f"Best Val Acc = {best_val_acc:.4f}")

    if not args.skip_test:
        test_list = read_data_from_txt(args.test_file)
        test_dataset = MultiModalDataset(test_list, args.data_dir, tokenizer, max_length=MAX_TEXT_LENGTH, is_test=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        model.load_state_dict(torch.load("best_model.pt", map_location=device))
        predictions = predict_test(model, test_loader, device)
        with open("test_prediction.txt", 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["guid", "tag"])
            for guid, pred_label in predictions:
                writer.writerow([guid, pred_label])
        print("Inference done! Predictions saved to test_prediction.txt")
    else:
        print("Skip testing. No test_prediction.txt generated.")

if __name__ == "__main__":
    main()
