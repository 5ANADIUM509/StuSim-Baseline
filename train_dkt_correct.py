import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --------- 小工具：时间解析（用于排序） ---------
def parse_time(s: str) -> Optional[float]:
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    fmts = ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt).timestamp()
        except Exception:
            pass
    return None

CHOICES = {"A", "B", "C", "D"}

def read_jsonl_group_by_user(path: str, max_lines: int = 0) -> Dict[int, List[dict]]:
    """
    读 jsonl，按 UserId 聚合。
    返回：user_data[uid] = [raw_record, ...]
    """
    user_data: Dict[int, List[dict]] = defaultdict(list)
    n_read = 0
    n_skip = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_read += 1
            if max_lines > 0 and n_read > max_lines:
                break
            try:
                obj = json.loads(line)
            except Exception:
                n_skip += 1
                continue

            uid = obj.get("UserId") or obj.get("user_id")
            qid = obj.get("question_id")
            sa = obj.get("Student_Answer") or obj.get("student_answer")
            ca = obj.get("CorrectAnswer") or obj.get("correct_answer")

            if uid is None or qid is None or sa is None or ca is None:
                n_skip += 1
                continue

            try:
                uid = int(uid)
                qid = int(qid)
            except Exception:
                n_skip += 1
                continue

            sa = str(sa).strip().upper()
            ca = str(ca).strip().upper()
            if sa not in CHOICES or ca not in CHOICES:
                n_skip += 1
                continue

            # 记录原始信息，后面统一处理
            user_data[uid].append(obj)

    print(f"[read] lines={n_read} | users={len(user_data)} | skipped={n_skip}")
    return user_data


def build_sequences(user_data: Dict[int, List[dict]]) -> Tuple[Dict[int, List[Tuple[int,int]]], Dict[int,int]]:
    """
    把每个学生的记录变成序列：
      seq = [(q_idx, r), ...]
    其中 r = 1(答对) / 0(答错)
    同时构建 question_id -> q_idx 的映射
    """
    # 收集所有 question_id
    all_qids = set()
    for recs in user_data.values():
        for r in recs:
            all_qids.add(int(r["question_id"]))
    qids_sorted = sorted(all_qids)
    qid2idx = {qid: i for i, qid in enumerate(qids_sorted)}

    seqs: Dict[int, List[Tuple[int,int]]] = {}

    for uid, recs in user_data.items():
        # 按时间排序（解析失败则用原顺序）
        def key_fn(x):
            t = x.get("DateAnswered") or x.get("date_answered") or ""
            ts = parse_time(t)
            return ts if ts is not None else float("inf")

        try:
            recs_sorted = sorted(recs, key=key_fn)
        except Exception:
            recs_sorted = recs

        seq = []
        for obj in recs_sorted:
            qid = int(obj["question_id"])
            sa = str(obj.get("Student_Answer") or obj.get("student_answer")).strip().upper()
            ca = str(obj.get("CorrectAnswer") or obj.get("correct_answer")).strip().upper()
            r = 1 if sa == ca else 0
            seq.append((qid2idx[qid], r))

        if len(seq) >= 2:  # 至少能预测下一步
            seqs[uid] = seq

    print(f"[seq] num_questions={len(qid2idx)} | usable_users={len(seqs)}")
    return seqs, qid2idx


def split_users(seqs: Dict[int, List[Tuple[int,int]]], seed: int = 0) -> Tuple[List[int], List[int], List[int]]:
    """
    按学生切分：train/val/test 学生不重叠
    """
    uids = list(seqs.keys())
    rng = random.Random(seed)
    rng.shuffle(uids)

    n = len(uids)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_u = uids[:n_train]
    val_u = uids[n_train:n_train + n_val]
    test_u = uids[n_train + n_val:]
    print(f"[split] users total={n} | train={len(train_u)} val={len(val_u)} test={len(test_u)}")
    return train_u, val_u, test_u


@dataclass
class SeqBatch:
    x: torch.Tensor          # [B, T]  interaction token ids
    q_next: torch.Tensor     # [B, T]  next question indices (target question)
    r_next: torch.Tensor     # [B, T]  next correctness (0/1)
    mask: torch.Tensor       # [B, T]  valid positions mask (bool)


class DKTDataset(Dataset):
    """
    把每个学生的序列用于 DKT 训练：
      输入 x_t = (q_t, r_t)
      目标是预测 (q_{t+1}, r_{t+1})
    也就是：长度 L 的序列会产生 L-1 个训练位置
    """
    def __init__(self, seqs: Dict[int, List[Tuple[int,int]]], user_ids: List[int], num_questions: int):
        self.num_questions = num_questions
        self.user_ids = user_ids
        self.seqs = seqs

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx: int):
        uid = self.user_ids[idx]
        seq = self.seqs[uid]  # [(q, r), ...], length L

        # 构造 interaction token: token = q + r * num_questions （大小 2Q）
        q = [qr[0] for qr in seq]
        r = [qr[1] for qr in seq]
        x = [q[t] + r[t] * self.num_questions for t in range(len(seq))]

        # 目标：预测下一步 (q_{t+1}, r_{t+1})
        # 所以有效位置是 t=0..L-2，对应目标是 index 1..L-1
        x_in = x[:-1]
        q_next = q[1:]
        r_next = r[1:]
        return {
            "x": torch.tensor(x_in, dtype=torch.long),          # [L-1]
            "q_next": torch.tensor(q_next, dtype=torch.long),   # [L-1]
            "r_next": torch.tensor(r_next, dtype=torch.float32) # [L-1]
        }


def collate_fn(batch: List[dict]) -> SeqBatch:
    """
    padding 到同长度：
      x: [B, T]
      q_next: [B, T]
      r_next: [B, T]
      mask: [B, T] 表示哪些位置有效
    """
    lens = [b["x"].shape[0] for b in batch]
    T = max(lens)
    B = len(batch)

    x = torch.zeros(B, T, dtype=torch.long)
    q_next = torch.zeros(B, T, dtype=torch.long)
    r_next = torch.zeros(B, T, dtype=torch.float32)
    mask = torch.zeros(B, T, dtype=torch.bool)

    for i, b in enumerate(batch):
        L = b["x"].shape[0]
        x[i, :L] = b["x"]
        q_next[i, :L] = b["q_next"]
        r_next[i, :L] = b["r_next"]
        mask[i, :L] = True

    return SeqBatch(x=x, q_next=q_next, r_next=r_next, mask=mask)


class DKTModel(nn.Module):
    """
    标准 DKT：
      - 输入：interaction token (q,r) -> embedding
      - LSTM 输出 hidden states h_t
      - 线性层输出对“所有题”的 logits：logits_t shape [B, T, Q]
      - 对每个位置 t，用 logits_t[ q_next ] 预测 r_next
    """
    def __init__(self, num_questions: int, emb_dim: int = 64, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.num_questions = num_questions
        self.vocab_size = 2 * num_questions

        self.emb = nn.Embedding(self.vocab_size, emb_dim)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, num_questions)  # 每个时间步输出 Q 个logit（每题是否答对）

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T] interaction token ids
        return logits: [B, T, Q]
        """
        e = self.emb(x)               # [B, T, emb_dim]
        h, _ = self.lstm(e)           # [B, T, hidden_dim]
        h = self.drop(h)
        logits = self.out(h)          # [B, T, Q]
        return logits


def batch_metrics_from_logits(logits: torch.Tensor, q_next: torch.Tensor, r_next: torch.Tensor, mask: torch.Tensor):
    """
    logits: [B, T, Q]
    q_next: [B, T]
    r_next: [B, T] float 0/1
    mask:   [B, T] bool
    取出每个位置对应 q_next 的 logit 作为预测，计算：
      - nll (BCE)
      - acc (阈值0.5)
      - auc（如果可算）
    """
    # 取出目标题目对应的 logits：gather
    # logits_target: [B, T]
    logits_target = logits.gather(dim=2, index=q_next.unsqueeze(-1)).squeeze(-1)

    # 只统计有效位置
    logits_flat = logits_target[mask]
    y_flat = r_next[mask]

    # BCE with logits
    loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
    loss = loss_fn(logits_flat, y_flat)

    prob = torch.sigmoid(logits_flat)
    pred = (prob >= 0.5).float()
    acc = (pred == y_flat).float().mean().item()

    # AUC：需要同时有正负样本
    auc = None
    y_np = y_flat.detach().cpu().numpy()
    p_np = prob.detach().cpu().numpy()
    if (y_np.sum() > 0) and (y_np.sum() < len(y_np)):
        # 手写一个简单AUC（避免依赖 sklearn）
        # AUC = rank-based: (sum ranks of positives - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
        order = np.argsort(p_np)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(p_np) + 1)
        n_pos = int(y_np.sum())
        n_neg = len(y_np) - n_pos
        sum_ranks_pos = ranks[y_np == 1].sum()
        auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg + 1e-12)
        auc = float(auc)

    return loss, acc, auc


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    losses = []
    accs = []
    aucs = []
    for batch in loader:
        x = batch.x.to(device)
        q_next = batch.q_next.to(device)
        r_next = batch.r_next.to(device)
        mask = batch.mask.to(device)

        logits = model(x)
        loss, acc, auc = batch_metrics_from_logits(logits, q_next, r_next, mask)
        losses.append(loss.item())
        accs.append(acc)
        if auc is not None:
            aucs.append(auc)

    out = {
        "nll": float(np.mean(losses)) if losses else math.nan,
        "acc": float(np.mean(accs)) if accs else math.nan,
        "auc": float(np.mean(aucs)) if aucs else math.nan,
    }
    return out


def train_one_epoch(model: nn.Module, loader: DataLoader, optim: torch.optim.Optimizer, device: torch.device):
    model.train()
    losses = []
    accs = []
    aucs = []

    for batch in loader:
        x = batch.x.to(device)
        q_next = batch.q_next.to(device)
        r_next = batch.r_next.to(device)
        mask = batch.mask.to(device)

        logits = model(x)
        loss, acc, auc = batch_metrics_from_logits(logits, q_next, r_next, mask)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optim.step()

        losses.append(loss.item())
        accs.append(acc)
        if auc is not None:
            aucs.append(auc)

    out = {
        "nll": float(np.mean(losses)) if losses else math.nan,
        "acc": float(np.mean(accs)) if accs else math.nan,
        "auc": float(np.mean(aucs)) if aucs else math.nan,
    }
    return out


def plot_curves(curves: dict, save_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(curves["train_acc"]) + 1))
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, curves["train_acc"], marker="o", label="train_acc")
    plt.plot(epochs, curves["val_acc"], marker="s", label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("DKT Correctness Prediction Accuracy vs Epoch")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--max_lines", type=int, default=0, help="0 = read all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--emb_dim", type=int, default=64)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--plot_path", type=str, default="dkt_correct_acc_curve.png")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    user_data = read_jsonl_group_by_user(args.data, max_lines=args.max_lines)
    seqs, qid2idx = build_sequences(user_data)

    train_u, val_u, test_u = split_users(seqs, seed=args.seed)

    train_ds = DKTDataset(seqs, train_u, num_questions=len(qid2idx))
    val_ds = DKTDataset(seqs, val_u, num_questions=len(qid2idx))
    test_ds = DKTDataset(seqs, test_u, num_questions=len(qid2idx))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_fn)

    print(f"[data] num_questions={len(qid2idx)} | users train/val/test = {len(train_u)}/{len(val_u)}/{len(test_u)}")

    model = DKTModel(num_questions=len(qid2idx), emb_dim=args.emb_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    curves = {"train_acc": [], "val_acc": []}
    best_val_acc = -1.0
    best_state = None

    for ep in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optim, device)
        va = evaluate(model, val_loader, device)

        curves["train_acc"].append(tr["acc"])
        curves["val_acc"].append(va["acc"])

        print(
            f"Epoch {ep:02d} | "
            f"train_nll={tr['nll']:.4f} train_acc={tr['acc']:.4f} train_auc={tr['auc'] if tr['auc'] is not None else 'nan'} | "
            f"val_nll={va['nll']:.4f} val_acc={va['acc']:.4f} val_auc={va['auc'] if va['auc'] is not None else 'nan'}"
        )

        # 早停/保存最佳
        if va["acc"] > best_val_acc:
            best_val_acc = va["acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    te = evaluate(model, test_loader, device)
    print("==== Test (best val_acc) ====")
    print(te)

    plot_curves(curves, args.plot_path)
    print(f"[saved] acc curve -> {args.plot_path}")


if __name__ == "__main__":
    main()
