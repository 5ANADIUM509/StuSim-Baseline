import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# 1) 工具：时间解析（用于按答题时间排序）
# -----------------------------
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


# -----------------------------
# 2) 读数据：jsonl -> 按用户聚合
# -----------------------------
def read_jsonl_group_by_user(path: str, max_lines: int = 0) -> Dict[int, List[dict]]:
    """
    读 jsonl，按 UserId 聚合
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

            user_data[uid].append(obj)

    print(f"[read] lines={n_read} | users={len(user_data)} | skipped={n_skip}")
    return user_data


# -----------------------------
# 3) 构建序列：(q_idx, r)
#    r=1 答对，r=0 答错
# -----------------------------
def build_full_sequences(user_data: Dict[int, List[dict]]) -> Tuple[Dict[int, List[Tuple[int, int]]], Dict[int, int]]:
    """
    返回：
      seqs_full[uid] = [(q_idx, r), ...]  # 使用所有题目
      qid2idx: question_id -> [0..Q-1]
    """
    all_qids = set()
    for recs in user_data.values():
        for r in recs:
            all_qids.add(int(r["question_id"]))
    qids_sorted = sorted(all_qids)
    qid2idx = {qid: i for i, qid in enumerate(qids_sorted)}

    seqs_full: Dict[int, List[Tuple[int, int]]] = {}

    for uid, recs in user_data.items():
        # 按 DateAnswered 排序（解析失败就放到最后）
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

        if len(seq) >= 2:  # 至少有一步可预测下一步
            seqs_full[uid] = seq

    print(f"[seq_full] num_questions={len(qid2idx)} | usable_users={len(seqs_full)}")
    return seqs_full, qid2idx


# -----------------------------
# 4) item split：按题目划分 train/val/test
# -----------------------------
def split_questions(num_questions: int, seed: int = 0, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """
    把题目 index [0..Q-1] 随机切成 train/val/test 三份
    """
    qidx = list(range(num_questions))
    rng = random.Random(seed)
    rng.shuffle(qidx)

    n = len(qidx)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    train_q = set(qidx[:n_train])
    val_q = set(qidx[n_train:n_train + n_val])
    test_q = set(qidx[n_train + n_val:])

    print(f"[item_split] Q={n} | trainQ={len(train_q)} valQ={len(val_q)} testQ={len(test_q)}")
    return train_q, val_q, test_q


# -----------------------------
# 5) 把 (q,r) 编成 token：token = q + r*Q （等价论文 one-hot 位置）
# -----------------------------
def encode_interaction_tokens(seq: List[Tuple[int, int]], Q: int) -> List[int]:
    # (q,r) -> [0..2Q-1]
    return [q + r * Q for (q, r) in seq]


# -----------------------------
# 6) Dataset：按用户给一条序列
#    - train：只用 trainQ 的交互（严格不看 val/test 题）
#    - val/test：用全序列，但只在目标题集上计分
# -----------------------------
@dataclass
class SeqBatch:
    x: torch.Tensor          # [B, T] 当前步输入 token（对应 x_t）
    q_next: torch.Tensor     # [B, T] 下一题 q_{t+1}
    r_next: torch.Tensor     # [B, T] 下一题是否答对 r_{t+1}
    pad_mask: torch.Tensor   # [B, T] padding mask（True=有效）
    tgt_mask: torch.Tensor   # [B, T] 是否属于目标题集（True=要计分）


class DKTSeqDataset(Dataset):
    """
    给定每个用户的序列 seq_full：
      输入 x_t = (q_t, r_t)
      预测下一步 (q_{t+1}, r_{t+1})
    这里我们还支持 “只在指定题集合 target_q 计分”
    """
    def __init__(
        self,
        seqs: Dict[int, List[Tuple[int, int]]],
        Q: int,
        target_q: Set[int],
        # history_mode:
        # - "full": 输入用全序列
        # - "train_only": 输入只用 train 序列（严格隔离新题）
        history_mode: str = "full",
        train_q: Optional[Set[int]] = None,
    ):
        self.uids = list(seqs.keys())
        self.seqs = seqs
        self.Q = Q
        self.target_q = target_q
        self.history_mode = history_mode
        self.train_q = train_q

        if history_mode == "train_only" and train_q is None:
            raise ValueError("history_mode=train_only 需要提供 train_q")

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx: int):
        uid = self.uids[idx]
        seq_full = self.seqs[uid]  # [(q,r), ...] length L

        # 决定用于 LSTM 输入的历史序列
        if self.history_mode == "full":
            seq_in = seq_full
        else:
            # 只保留 trainQ 的交互作为历史输入（严格：训练没见过的新题不会出现在输入里）
            seq_in = [(q, r) for (q, r) in seq_full if q in self.train_q]

        # 注意：为了预测下一步，至少需要2条
        if len(seq_in) < 2:
            # 兜底：返回一个空样本，后面 collate 时会跳过（更简单）
            return None

        q = [qr[0] for qr in seq_in]
        r = [qr[1] for qr in seq_in]
        x_tokens = encode_interaction_tokens(seq_in, self.Q)

        # 输入是前 L-1 步，目标是后 L-1 步
        x = x_tokens[:-1]
        q_next = q[1:]
        r_next = r[1:]

        # tgt_mask：只对“目标题集”里的 q_next 计分
        tgt_mask = [1 if (qq in self.target_q) else 0 for qq in q_next]

        return {
            "x": torch.tensor(x, dtype=torch.long),                 # [T]
            "q_next": torch.tensor(q_next, dtype=torch.long),       # [T]
            "r_next": torch.tensor(r_next, dtype=torch.float32),    # [T]
            "tgt_mask": torch.tensor(tgt_mask, dtype=torch.bool),   # [T]
        }


def collate_fn(batch: List[Optional[dict]]) -> SeqBatch:
    # 过滤掉 None（序列太短的用户）
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        # 返回一个空 batch（上层要处理）
        return SeqBatch(
            x=torch.zeros(0, 0, dtype=torch.long),
            q_next=torch.zeros(0, 0, dtype=torch.long),
            r_next=torch.zeros(0, 0, dtype=torch.float32),
            pad_mask=torch.zeros(0, 0, dtype=torch.bool),
            tgt_mask=torch.zeros(0, 0, dtype=torch.bool),
        )

    lens = [b["x"].shape[0] for b in batch]
    T = max(lens)
    B = len(batch)

    x = torch.zeros(B, T, dtype=torch.long)
    q_next = torch.zeros(B, T, dtype=torch.long)
    r_next = torch.zeros(B, T, dtype=torch.float32)
    pad_mask = torch.zeros(B, T, dtype=torch.bool)
    tgt_mask = torch.zeros(B, T, dtype=torch.bool)

    for i, b in enumerate(batch):
        L = b["x"].shape[0]
        x[i, :L] = b["x"]
        q_next[i, :L] = b["q_next"]
        r_next[i, :L] = b["r_next"]
        pad_mask[i, :L] = True
        tgt_mask[i, :L] = b["tgt_mask"]

    return SeqBatch(x=x, q_next=q_next, r_next=r_next, pad_mask=pad_mask, tgt_mask=tgt_mask)


# -----------------------------
# 7) 模型：标准 DKT
# -----------------------------
class DKTModel(nn.Module):
    """
    输入 token (q,r) -> Embedding -> LSTM -> 预测每个题的“答对 logit”
    logits: [B, T, Q]
    """
    def __init__(self, Q: int, emb_dim: int = 64, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.Q = Q
        self.vocab = 2 * Q

        self.emb = nn.Embedding(self.vocab, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, Q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.emb(x)          # [B,T,emb]
        h, _ = self.lstm(e)      # [B,T,H]
        h = self.drop(h)
        logits = self.out(h)     # [B,T,Q]
        return logits


# -----------------------------
# 8) 指标：acc / nll / auc（全局AUC，不按batch平均）
# -----------------------------
def auc_rank(prob: np.ndarray, y: np.ndarray) -> float:
    """
    纯 numpy 的 ROC AUC（rank-based），不依赖 sklearn
    """
    y = y.astype(np.int32)
    n = len(y)
    if n == 0:
        return float("nan")
    n_pos = int(y.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(prob)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)

    sum_ranks_pos = ranks[y == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg + 1e-12)
    return float(auc)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="mean")

    all_prob = []
    all_y = []
    losses = []
    hit = 0
    total = 0

    for batch in loader:
        if batch.x.numel() == 0:
            continue

        x = batch.x.to(device)
        q_next = batch.q_next.to(device)
        r_next = batch.r_next.to(device)
        valid = (batch.pad_mask & batch.tgt_mask).to(device)  # 只在目标题集位置计分

        logits = model(x)  # [B,T,Q]
        logits_t = logits.gather(2, q_next.unsqueeze(-1)).squeeze(-1)  # [B,T]

        lt = logits_t[valid]
        y = r_next[valid]

        if lt.numel() == 0:
            continue

        loss = bce(lt, y)
        losses.append(loss.item())

        prob = torch.sigmoid(lt)
        pred = (prob >= 0.5).float()
        hit += (pred == y).sum().item()
        total += y.numel()

        all_prob.append(prob.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())

    prob_np = np.concatenate(all_prob) if all_prob else np.array([])
    y_np = np.concatenate(all_y) if all_y else np.array([])

    out = {
        "nll": float(np.mean(losses)) if losses else float("nan"),
        "acc": hit / max(1, total),
        "auc": auc_rank(prob_np, y_np) if len(y_np) > 0 else float("nan"),
        "n": float(total),
    }
    return out


def train_one_epoch(model: nn.Module, loader: DataLoader, optim: torch.optim.Optimizer, device: torch.device) -> Dict[str, float]:
    model.train()
    bce = nn.BCEWithLogitsLoss(reduction="mean")

    all_prob = []
    all_y = []
    losses = []
    hit = 0
    total = 0

    for batch in loader:
        if batch.x.numel() == 0:
            continue

        x = batch.x.to(device)
        q_next = batch.q_next.to(device)
        r_next = batch.r_next.to(device)
        valid = (batch.pad_mask & batch.tgt_mask).to(device)

        logits = model(x)
        logits_t = logits.gather(2, q_next.unsqueeze(-1)).squeeze(-1)

        lt = logits_t[valid]
        y = r_next[valid]

        if lt.numel() == 0:
            continue

        loss = bce(lt, y)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optim.step()

        losses.append(loss.item())

        prob = torch.sigmoid(lt)
        pred = (prob >= 0.5).float()
        hit += (pred == y).sum().item()
        total += y.numel()

        all_prob.append(prob.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())

    prob_np = np.concatenate(all_prob) if all_prob else np.array([])
    y_np = np.concatenate(all_y) if all_y else np.array([])

    out = {
        "nll": float(np.mean(losses)) if losses else float("nan"),
        "acc": hit / max(1, total),
        "auc": auc_rank(prob_np, y_np) if len(y_np) > 0 else float("nan"),
        "n": float(total),
    }
    return out


# -----------------------------
# 9) 画曲线：acc / auc
# -----------------------------
def plot_curves(curves: dict, acc_path: str, auc_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(curves["train_acc"]) + 1))

    # ACC
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, curves["train_acc"], marker="o", label="train_acc (trainQ)")
    plt.plot(epochs, curves["val_acc"], marker="s", label="val_acc (valQ)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Item-split DKT Correctness: Accuracy vs Epoch")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_path, dpi=200)
    plt.close()

    # AUC
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, curves["train_auc"], marker="o", label="train_auc (trainQ)")
    plt.plot(epochs, curves["val_auc"], marker="s", label="val_auc (valQ)")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("Item-split DKT Correctness: AUC vs Epoch")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(auc_path, dpi=200)
    plt.close()


# -----------------------------
# 10) main
# -----------------------------
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

    # item split 比例
    ap.add_argument("--train_q_ratio", type=float, default=0.8)
    ap.add_argument("--val_q_ratio", type=float, default=0.1)

    # 评估时历史怎么喂：
    # - full：用全序列（包含新题的交互，但这些题embedding是未训练的）
    # - train_only：只用训练题交互作为历史（更“干净”，但更苛刻/更短历史）
    ap.add_argument("--eval_history", type=str, default="full", choices=["full", "train_only"])

    ap.add_argument("--acc_curve_path", type=str, default="item_split_acc_curve.png")
    ap.add_argument("--auc_curve_path", type=str, default="item_split_auc_curve.png")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    user_data = read_jsonl_group_by_user(args.data, max_lines=args.max_lines)
    seqs_full, qid2idx = build_full_sequences(user_data)
    Q = len(qid2idx)

    train_q, val_q, test_q = split_questions(Q, seed=args.seed, train_ratio=args.train_q_ratio, val_ratio=args.val_q_ratio)

    # ---------- 训练序列：严格过滤掉 val/test 题 ----------
    seqs_train_only: Dict[int, List[Tuple[int, int]]] = {}
    dropped_users = 0
    for uid, seq in seqs_full.items():
        seq_tr = [(q, r) for (q, r) in seq if q in train_q]
        if len(seq_tr) >= 2:
            seqs_train_only[uid] = seq_tr
        else:
            dropped_users += 1
    print(f"[train_seq] users={len(seqs_train_only)} | dropped(short after filter)={dropped_users}")

    # DataLoaders
    train_ds = DKTSeqDataset(
        seqs=seqs_train_only,
        Q=Q,
        target_q=train_q,           # 训练只在 trainQ 计分
        history_mode="full",        # 训练数据本身已经过滤到 trainQ 了，这里 full 就等于 train-only
        train_q=train_q,
    )

    # val/test：用全序列，但只在目标题集位置计分
    # 注意：eval_history 影响“输入历史”是否也过滤到 trainQ（避免输入出现新题）
    val_ds = DKTSeqDataset(
        seqs=seqs_full,
        Q=Q,
        target_q=val_q,
        history_mode=args.eval_history,
        train_q=train_q,
    )
    test_ds = DKTSeqDataset(
        seqs=seqs_full,
        Q=Q,
        target_q=test_q,
        history_mode=args.eval_history,
        train_q=train_q,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_fn)

    print(f"[data] Q={Q} | trainQ={len(train_q)} valQ={len(val_q)} testQ={len(test_q)} | eval_history={args.eval_history}")

    model = DKTModel(Q=Q, emb_dim=args.emb_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    curves = {"train_acc": [], "val_acc": [], "train_auc": [], "val_auc": []}
    best_val_auc = -1.0
    best_state = None

    for ep in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optim, device)
        va = evaluate(model, val_loader, device)

        curves["train_acc"].append(tr["acc"])
        curves["val_acc"].append(va["acc"])
        curves["train_auc"].append(tr["auc"])
        curves["val_auc"].append(va["auc"])

        print(
            f"Epoch {ep:02d} | "
            f"train(n={int(tr['n'])}) nll={tr['nll']:.4f} acc={tr['acc']:.4f} auc={tr['auc']:.4f} | "
            f"val(n={int(va['n'])}) nll={va['nll']:.4f} acc={va['acc']:.4f} auc={va['auc']:.4f}"
        )

        if not np.isnan(va["auc"]) and va["auc"] > best_val_auc:
            best_val_auc = va["auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    te = evaluate(model, test_loader, device)
    print("==== Test (best val_auc) ====")
    print(te)

    plot_curves(curves, args.acc_curve_path, args.auc_curve_path)
    print(f"[saved] acc curve -> {args.acc_curve_path}")
    print(f"[saved] auc curve -> {args.auc_curve_path}")


if __name__ == "__main__":
    main()
