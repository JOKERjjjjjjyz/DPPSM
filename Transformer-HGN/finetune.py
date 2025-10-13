#!/usr/bin/env python
"""
Fine-tune PassBERT on a password-probability regression task
============================================================

Input TSV: ``password \t frequency``
-----------------------------------
* **password** 原始密码字符串。
* **frequency** [0, 1] 或正整数出现次数。脚本会自动归一化到概率域。

Workflow
--------
1. 载入预训练权重（MLM 阶段得到的 ``passbert.pt``）。
2. 新增一个 **回归头**：取 `[CLS]` 位置的 256 维向量 → Linear(256→1)。
3. 全参数更新（encoder 与回归头都会被训练）。
4. 使用 ``BCEWithLogitsLoss``（默认）或 ``MSELoss``，可通过参数切换。

Usage
-----
::

    python passbert_finetune.py \
        --train pwd_freq.tsv \
        --pretrained passbert.pt \
        --epochs 10 \
        --batch 1024 \
        --lr 1e-4 \
        --loss bce 

Requirements 与预训练脚本相同。
"""

import argparse
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


import json
# ----------------------------- 共享的 tokenizer ----------------------------- #
SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[SEP]", "[MASK]"]
ASCII_PRINTABLE = [chr(i) for i in range(32, 127)]
VOCAB = SPECIAL_TOKENS + ASCII_PRINTABLE
VOCAB_SIZE = len(VOCAB)
char2idx = {ch: i for i, ch in enumerate(VOCAB)}
PAD_IDX, CLS_IDX, SEP_IDX, MASK_IDX = (char2idx[t] for t in SPECIAL_TOKENS)
MAX_LEN = 34


def encode_password(pwd: str) -> List[int]:
    core = list(pwd)[:32]
    tokens = [CLS_IDX] + [char2idx.get(c, char2idx["?"]) for c in core] + [SEP_IDX]
    tokens += [PAD_IDX] * (MAX_LEN - len(tokens))
    return tokens[:MAX_LEN]

# ----------------------------- Dataset ------------------------------------- #

class PwdFreqDataset(Dataset):
    """TSV: password \t frequency"""

    def __init__(self, path: str):
        self.pairs = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            pwd, freq = line.split("\t")
            freq = float(freq)
            self.pairs.append((pwd, freq))
        # 归一化到 0~1 概率
        freqs = np.array([f for _, f in self.pairs], dtype=np.float32)
        probs = freqs / freqs.max()
        self.pairs = [(pwd, prob) for (pwd, _), prob in zip(self.pairs, probs)]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pwd, prob = self.pairs[idx]
        tokens = torch.tensor(encode_password(pwd), dtype=torch.long)
        label = torch.tensor(prob, dtype=torch.float32)
        return tokens, label

# ----------------------------- Model --------------------------------------- #

class PassBERT(nn.Module):
    def __init__(self, hidden_dim=256, ffn_dim=512, n_heads=8, n_layers=4):
        super().__init__()
        self.char_emb = nn.Embedding(VOCAB_SIZE, hidden_dim, padding_idx=PAD_IDX)
        self.pos_emb = nn.Embedding(MAX_LEN, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(hidden_dim, n_heads, ffn_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)

    def forward(self, inp):  # [B, L]
        pos_ids = torch.arange(inp.size(1), device=inp.device).unsqueeze(0).expand_as(inp)
        x = self.char_emb(inp) + self.pos_emb(pos_ids)
        x = self.encoder(x)  # [B, L, H]
        cls_vec = x[:, 0]     # 取 [CLS] 向量
        return cls_vec        # [B, H]


class PasswordRegressor(nn.Module):
    def __init__(self, base_encoder: PassBERT):
        super().__init__()
        self.encoder = base_encoder
        self.head = nn.Linear(256, 1)  # 256 → 概率 logits

    def forward(self, inp):
        h = self.encoder(inp)
        logits = self.head(h).squeeze(-1)  # [B]
        return logits

# ----------------------------- Training ------------------------------------ #


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PwdFreqDataset(args.train_tsv)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, pin_memory=True)

    # base encoder
    encoder = PassBERT()
    if args.pretrained:
        sd = torch.load(args.pretrained, map_location="cpu")
        encoder.load_state_dict(sd, strict=False)
        print(f"Loaded pretrained weights from {args.pretrained}")
    model = PasswordRegressor(encoder).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if args.loss_type == "bce":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        sum_loss = 0.0
        for tokens, labels in tqdm(loader, desc=f"Epoch {epoch}"):
            tokens = tokens.to(device)
            labels = labels.to(device)
            logits = model(tokens)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.item() * tokens.size(0)
            total += tokens.size(0)
        print(f"Epoch {epoch}: loss = {sum_loss / total:.4f}")

    torch.save(model.state_dict(), args.out)
    print(f"Finetuned model saved to {args.out}")

# ----------------------------- CLI ----------------------------------------- #


# def parse_args():
#     p = argparse.ArgumentParser(description="Finetune PassBERT for password frequency regression")
#     p.add_argument("--train", required=True, help="TSV file: password<TAB>frequency")
#     p.add_argument("--pretrained", required=True, help="path to passbert.pt (MLM pretrain)")
#     p.add_argument("--epochs", type=int, default=10)
#     p.add_argument("--batch", type=int, default=1024)
#     p.add_argument("--lr", type=float, default=1e-4)
#     p.add_argument("--loss", choices=["bce", "mse"], default="bce", help="loss type")
#     p.add_argument("--out", default="passbert_finetuned.pt")
#     return p.parse_args()


# if __name__ == "__main__":
#     args = parse_args()
#     train(args)


def parse_args():
    p = argparse.ArgumentParser(description="Finetune PassBERT for password frequency regression")
    # 支持一个 --config 指定 JSON 文件
    p.add_argument("--config", type=str, help="Path to JSON config file")

    # 以下都是默认值，仅在命令行没有用 --config 覆盖时有效
    p.add_argument("--train_tsv",    type=str, default=None,
                   help="TSV 文件：password<TAB>frequency")
    p.add_argument("--pretrained",   type=str, default=None,
                   help="预训练模型 passbert.pt 的路径")
    p.add_argument("--epochs",       type=int, default=10,   help="训练轮数")
    p.add_argument("--batch",   type=int, default=1024, help="batch size")
    p.add_argument("--lr",type=float, default=1e-4,help="学习率")
    p.add_argument("--loss_type",    type=str, choices=["bce", "mse"], default="bce",
                   help="损失函数类型")
    p.add_argument("--out", type=str, default="passbert_finetuned.pt",
                   help="微调后模型保存路径")

    # 同步模型结构超参，必须与预训练脚本里的一致
    p.add_argument("--hidden_dim", type=int, default=256, help="隐藏层维度")
    p.add_argument("--ffn_dim",    type=int, default=512, help="前馈层维度")
    p.add_argument("--num_heads",  type=int, default=8, help="Attention head 数")
    p.add_argument("--num_layers", type=int, default=4, help="Transformer 层数")

    # 是否全参数更新；false 表示冻结 encoder 只训练 head
    p.add_argument("--finetune_full", action="store_true",
                   help="如果有此 flag，则全参数更新，否则只训练回归 Head")

    args = p.parse_args()

    # 有 --config 时，用 JSON 中的值覆盖 args
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config_args = json.load(f)
        for key, value in config_args.items():
            setattr(args, key, value)

    # train_tsv 和 pretrained 是必填项
    if not args.train_tsv or not args.pretrained:
        raise ValueError("必须通过 --train_tsv、--pretrained 或在 JSON 中指定相应字段")
    return args

if __name__ == "__main__":
    args = parse_args()
    train(args)