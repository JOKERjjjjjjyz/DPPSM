#!/usr/bin/env python
"""
PassBERT‑style character‑level MLM pre‑training script
=====================================================

*   **Task**   Masked‑Language‑Modeling (character level) on password strings
*   **Paper spec** 4 encoder blocks, hidden 256, FFN 512, vocab 99, max‑len 34
*   **Data flow**
        raw_passwords.txt  ──►  (on‑the‑fly pivot generator)  ──►  passwords_mlm.txt
        passwords_mlm.txt  ──►  DataLoader  ──►  PassBERT  ──►  CrossEntropyLoss

Usage
-----
::

    python passbert_mlm_pretrain.py --train passwords.txt \
                                    --epochs 5 \
                                    --batch 512 \
                                    --lr 3e-4

The script will
1.  Look for ``passwords.txt_mlm.txt``.  If it does not exist, generate it
    with the **pivot rule** described in the paper but using ``len(password)``
    pivots per sample instead of a fixed 20.
2.  Train the model with standard character‑level MLM (only masked tokens
    contribute to the loss).
3.  Save model parameters to ``passbert.pt`` (path can be changed with
   ``--out``).

Requirements
------------
* Python >= 3.8
* PyTorch >= 2.0
* tqdm
* numpy

Install with ::

    pip install torch tqdm numpy
"""

import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 1. Vocabulary ----------------------------------------------------------------
# 95 printable ASCII + 4 specials = 99 tokens
SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[SEP]", "[MASK]"]  # indices 0‑3
ASCII_PRINTABLE = [chr(i) for i in range(32, 127)]  # 95 chars (space … ~)
VOCAB: List[str] = SPECIAL_TOKENS + ASCII_PRINTABLE  # len == 99
VOCAB_SIZE = len(VOCAB)

char2idx = {ch: i for i, ch in enumerate(VOCAB)}
idx2char = {i: ch for i, ch in enumerate(VOCAB)}

PAD_IDX, CLS_IDX, SEP_IDX, MASK_IDX = (char2idx[t] for t in SPECIAL_TOKENS)

MAX_LEN = 34  # [CLS] + 32 chars + [SEP]
MASKING_RATE = 0.15  # 15 % as in BERT

# ---------------------------------------------------------------------------
# 2. Tokenisation helpers -----------------------------------------------------


def encode_password(pwd: str) -> List[int]:
    """Convert raw password string to list of token indices (len == MAX_LEN)."""
    # Truncate to 32 chars between CLS and SEP
    core = list(pwd)[:32]
    tokens = [CLS_IDX] + [char2idx.get(c, char2idx["?"]) for c in core] + [SEP_IDX]
    tokens += [PAD_IDX] * (MAX_LEN - len(tokens))
    return tokens[:MAX_LEN]


# ---------------------------------------------------------------------------
# 3. Pivot (masked sequence) generation --------------------------------------


def make_single_pivot(tokens: List[int]) -> Tuple[List[int], List[int]]:
    """Return (pivot_tokens, target_tokens).

    * 15 % of *character* tokens (not specials / pads) are corrupted.
    * 80 % → [MASK]  |  10 % → random char  |  10 % → unchanged
    """
    pivot = tokens.copy()
    target = tokens.copy()  # ground‑truth sequence

    # candidate positions are real character slots: 1 … (first PAD or SEP‑1)
    valid_pos = [i for i, t in enumerate(tokens)
                 if 0 < i < MAX_LEN - 1 and t not in (PAD_IDX, SEP_IDX)]
    num_to_mask = max(1, int(round(len(valid_pos) * MASKING_RATE)))
    random.shuffle(valid_pos)
    masked = valid_pos[:num_to_mask]

    for pos in masked:
        r = random.random()
        if r < 0.8:               # 80 % [MASK]
            pivot[pos] = MASK_IDX
        elif r < 0.9:             # 10 % random char (not special)
            pivot[pos] = random.randint(len(SPECIAL_TOKENS), VOCAB_SIZE - 1)
        else:                     # 10 % unchanged
            pass
    return pivot, target


def generate_pivots_for_password(pwd: str) -> List[Tuple[List[int], List[int]]]:
    tokens = encode_password(pwd)
    n_pivots = max(1, len(pwd))  # paper uses 20; we use |pwd|
    pivots = [make_single_pivot(tokens) for _ in range(n_pivots)]
    return pivots


# ---------------------------------------------------------------------------
# 4. Dataset preparation ------------------------------------------------------


def build_mlm_dataset(src_path: str, dst_path: str):
    """Create <pivot \t target> TSV lines and write to *dst_path*."""
    out = Path(dst_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(src_path, "r", encoding="utf-8") as fr, \
            open(out, "w", encoding="utf-8") as fw:
        for line in fr:
            pwd = line.rstrip("\n")
            for pivot, target in generate_pivots_for_password(pwd):
                fw.write(",".join(map(str, pivot)))
                fw.write("\t")
                fw.write(",".join(map(str, target)))
                fw.write("\n")


class MLMDataset(Dataset):
    def __init__(self, tsv_path: str):
        self.samples = Path(tsv_path).read_text(encoding="utf-8").strip().split("\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pivot_str, target_str = self.samples[idx].split("\t")
        pivot = torch.tensor([int(x) for x in pivot_str.split(",")], dtype=torch.long)
        target = torch.tensor([int(x) for x in target_str.split(",")], dtype=torch.long)
        # create label tensor: only masked positions contribute, others set to -100
        labels = target.clone()
        mask_positions = pivot == MASK_IDX
        labels[~mask_positions] = -100  # ignored in CrossEntropyLoss
        return pivot, labels


# ---------------------------------------------------------------------------
# 5. Model definition ---------------------------------------------------------


class PassBERT(nn.Module):
    def __init__(self,
                 vocab_size: int = VOCAB_SIZE,
                 hidden_dim: int = 256,
                 ffn_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 max_len: int = MAX_LEN):
        super().__init__()
        self.char_emb = nn.Embedding(vocab_size, hidden_dim, padding_idx=PAD_IDX)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim,
                                               nhead=num_heads,
                                               dim_feedforward=ffn_dim,
                                               batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.fc = nn.Linear(hidden_dim, hidden_dim)       
        self.out_proj = nn.Linear(hidden_dim, vocab_size)


    def forward(self, inp: torch.Tensor):  # inp: [B, L]
        pos_ids = torch.arange(inp.size(1), device=inp.device).unsqueeze(0).expand_as(inp)
        x = self.char_emb(inp) + self.pos_emb(pos_ids)
        x = self.encoder(x)  # still [B, L, H] because batch_first=True
        logits = self.fc(x)  
        logits = self.out_proj(logits)
        return logits  # [B, L, vocab]


# ---------------------------------------------------------------------------
# 6. Training loop ------------------------------------------------------------


def train(args):
    src_path = args.train
    mlm_path = src_path + "_mlm.txt"

    if not Path(mlm_path).exists():
        print(f"[Parser] generating MLM dataset: {mlm_path}")
        build_mlm_dataset(src_path, mlm_path)
    else:
        print(f"[Parser] found existing MLM dataset: {mlm_path}")

    dataset = MLMDataset(mlm_path)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PassBERT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for pivots, labels in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}"):
            pivots = pivots.to(device)
            labels = labels.to(device)
            logits = model(pivots)
            loss = criterion(logits.view(-1, VOCAB_SIZE), labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * pivots.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}: average loss = {avg_loss:.4f}")

    torch.save(model.state_dict(), args.out)
    print(f"Model saved to {args.out}")


# ---------------------------------------------------------------------------
# 7. CLI ----------------------------------------------------------------------


# def parse_args():
#     p = argparse.ArgumentParser(description="PassBERT MLM pre‑training")
#     p.add_argument("--train", required=True, help="raw password .txt file (one per line)")
#     p.add_argument("--epochs", type=int, default=5)
#     p.add_argument("--batch", type=int, default=512)
#     p.add_argument("--lr", type=float, default=3e-4)
#     p.add_argument("--out", default="passbert.pt")
#     return p.parse_args()
def parse_args():
    import json

    p = argparse.ArgumentParser(description="PassBERT MLM pre‑training")
    p.add_argument("--config", type=str, help="Path to JSON config file")

    # Provide defaults in case config not used
    p.add_argument("--train", type=str, default=None)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--out", type=str, default="passbert.pt")
    args = p.parse_args()

    # Override using config if provided
    if args.config:
        with open(args.config, "r") as f:
            config_args = json.load(f)
        for key, value in config_args.items():
            setattr(args, key, value)

    if not args.train:
        raise ValueError("Training file must be specified via --train or config JSON.")

    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
