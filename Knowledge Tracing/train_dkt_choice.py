# train_dkt_choice_plot.py
# ------------------------------------------------------------
# 用 DKT(LSTM) 做选择题 Option Tracing 基线：
# 输入：每个学生最近 window=30 次交互（题号 + 学生选项 + 可选是否做对）
# 条件：下一题题号 q_next 已知
# 输出：预测学生在下一题会选 A/B/C/D 哪个（4分类）
#
# 指标：Accuracy / Top-2 Accuracy（真实选项在模型预测概率最高的前2个里） / NLL(CrossEntropy)
# 评估：严格按学生切分 train/val/test（避免数据泄漏）
# 额外：保存 Accuracy 随 epoch 变化的曲线图（png）
# ------------------------------------------------------------

import json
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt


# -----------------------
# 固定映射：选项 -> 0/1/2/3
# -----------------------
CHOICE2IDX = {"A": 0, "B": 1, "C": 2, "D": 3}


def parse_time(s: str) -> float:
    """
    把 "2019-12-03 19:07:00.000" 解析成时间戳（用于排序）
    """
    dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")
    return dt.timestamp()


def parse_is_correct(obj: dict) -> int:
    """
    返回 0/1：这条记录是否做对

    兼容两种情况：
    1) 有 IsCorrect 字段，值是 "正确"/"错误"
    2) 用 CorrectAnswer 和 Student_Answer 比较
    """
    if "IsCorrect" in obj and isinstance(obj["IsCorrect"], str):
        return 1 if obj["IsCorrect"].strip() == "正确" else 0

    ca = str(obj.get("CorrectAnswer", "")).strip().upper()
    sa = str(obj.get("Student_Answer", "")).strip().upper()
    if ca in CHOICE2IDX and sa in CHOICE2IDX:
        return 1 if ca == sa else 0

    # 如果缺字段，默认0（你也可以改成：缺字段就跳过该条）
    return 0


@dataclass
class Sample:
    """
    一个训练样本：用 window 步历史 -> 预测下一步
    """
    uid: int
    hist_tokens: List[int]   # 长度=window
    q_next: int              # 下一题题号索引 [0..Q-1]
    a_next: int              # 下一题真实选项 {0,1,2,3}


class KTWindowDataset(Dataset):
    def __init__(self, samples: List[Sample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return (
            torch.tensor(s.hist_tokens, dtype=torch.long),  # [L]
            torch.tensor(s.q_next, dtype=torch.long),       # []
            torch.tensor(s.a_next, dtype=torch.long),       # []
        )


class DKTChoiceModel(nn.Module):
    """
    DKT(LSTM) 版本的选项预测模型：
    - 用 LSTM 从历史交互序列压缩出“学生状态” h
    - 再结合下一题题号 embedding，输出 4 个选项的 logits
    """
    def __init__(
        self,
        vocab_size: int,      # token 词表大小：Q*4 或 Q*8
        num_questions: int,   # Q
        emb_dim: int = 64,
        lstm_hidden: int = 128,
        q_emb_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 把每一步交互 token 变成向量
        self.inter_emb = nn.Embedding(vocab_size, emb_dim)

        # LSTM 读序列
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=lstm_hidden, batch_first=True)

        # 下一题题号 embedding（告诉模型“你要预测的是哪道题”）
        self.q_emb = nn.Embedding(num_questions, q_emb_dim)

        # 学生状态 + 下一题向量 -> 输出4类
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden + q_emb_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 4),
        )

    def forward(self, hist_tokens: torch.Tensor, q_next: torch.Tensor) -> torch.Tensor:
        """
        hist_tokens: [B, L]
        q_next:      [B]
        return logits: [B, 4]
        """
        x = self.inter_emb(hist_tokens)     # [B, L, emb_dim]
        _, (h_n, _) = self.lstm(x)          # h_n: [1, B, hidden]
        h = h_n[-1]                         # [B, hidden]
        qv = self.q_emb(q_next)             # [B, q_emb_dim]
        feat = torch.cat([h, qv], dim=-1)   # [B, hidden+q_emb_dim]
        logits = self.mlp(feat)             # [B, 4]
        return logits


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    """
    在 val/test 上评估：
    - acc：预测选项准确率
    - top2_acc：Top-2 准确率
    - nll：平均交叉熵（越小越好）
    """
    model.eval()
    ce_sum = nn.CrossEntropyLoss(reduction="sum")

    total = 0
    correct = 0
    top2 = 0
    total_loss = 0.0

    for hist_tokens, q_next, a_next in loader:
        hist_tokens = hist_tokens.to(device)
        q_next = q_next.to(device)
        a_next = a_next.to(device)

        logits = model(hist_tokens, q_next)      # [B, 4]
        total_loss += ce_sum(logits, a_next).item()

        pred = logits.argmax(dim=-1)
        correct += (pred == a_next).sum().item()

        top2_pred = logits.topk(k=2, dim=-1).indices
        top2 += (top2_pred == a_next.unsqueeze(-1)).any(dim=-1).sum().item()

        total += a_next.size(0)

    return {
        "acc": correct / max(total, 1),
        "top2_acc": top2 / max(total, 1),
        "nll": total_loss / max(total, 1),
        "n": total,
    }


def build_samples(
    jsonl_path: str,
    window: int = 30,
    use_correct: int = 1,
    max_lines: int = 0
) -> Tuple[List[Sample], Dict[int, int]]:
    """
    从 jsonl 构建样本：
      1) 读入所有记录（可限制 max_lines）
      2) 按 UserId 聚合
      3) 按 DateAnswered 排序
      4) 滑窗：用 window 条历史预测下一条

    token 编码：
    - use_correct=0: token = q_idx*4 + choice  （choice in 0..3）
    - use_correct=1: token = q_idx*8 + (choice*2 + is_correct)  （0..7）
      这样能把“过去做对/做错”也编码进历史（更像知识追踪）
    """
    per_user: Dict[int, List[Tuple[float, int, int, int]]] = {}
    all_qids = set()

    n_read = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            n_read += 1
            if max_lines > 0 and n_read > max_lines:
                break

            obj = json.loads(line)

            # 取必要字段
            uid = int(obj["UserId"])
            qid = int(obj["question_id"])
            ans = str(obj["Student_Answer"]).strip().upper()
            if ans not in CHOICE2IDX:
                continue

            a = CHOICE2IDX[ans]
            t = parse_time(obj["DateAnswered"])
            ic = parse_is_correct(obj)  # 0/1

            all_qids.add(qid)
            per_user.setdefault(uid, []).append((t, qid, a, ic))

    # 映射 question_id -> 连续索引
    qids_sorted = sorted(all_qids)
    qid2idx = {qid: i for i, qid in enumerate(qids_sorted)}

    samples: List[Sample] = []
    for uid, seq in per_user.items():
        # 按时间排序（很重要）
        seq.sort(key=lambda x: x[0])

        # 映射为 (q_idx, choice, is_correct)
        qa = [(qid2idx[qid], a, ic) for _, qid, a, ic in seq]

        # 至少 window+1 才能产生一个样本
        if len(qa) < window + 1:
            continue

        # stride=1 滑窗：一直滑到末尾
        for end in range(window, len(qa)):
            hist = qa[end - window:end]   # window 步历史
            q_next, a_next, _ = qa[end]   # 监督目标是下一步选项

            if use_correct:
                # (choice, is_correct) -> 0..7
                hist_tokens = [q * 8 + (a * 2 + ic) for (q, a, ic) in hist]
            else:
                hist_tokens = [q * 4 + a for (q, a, ic) in hist]

            samples.append(Sample(uid=uid, hist_tokens=hist_tokens, q_next=q_next, a_next=a_next))

    return samples, qid2idx


def split_users(samples: List[Sample], seed: int = 0) -> Tuple[List[int], List[int], List[int]]:
    """
    严格按学生切分：train/val/test 学生不重叠
    返回 train/val/test 三个“样本索引列表”
    """
    # uid -> 样本索引列表
    per_user_samples: Dict[int, List[int]] = {}
    for i, s in enumerate(samples):
        per_user_samples.setdefault(s.uid, []).append(i)

    uids = list(per_user_samples.keys())
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(uids), generator=g).tolist()
    uids = [uids[i] for i in perm]

    n = len(uids)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train_u = set(uids[:n_train])
    val_u = set(uids[n_train:n_train + n_val])
    test_u = set(uids[n_train + n_val:])

    train_idx, val_idx, test_idx = [], [], []
    for uid, idxs in per_user_samples.items():
        if uid in train_u:
            train_idx.extend(idxs)
        elif uid in val_u:
            val_idx.extend(idxs)
        elif uid in test_u:
            test_idx.extend(idxs)

    return train_idx, val_idx, test_idx


def plot_acc_curve(train_acc: List[float], val_acc: List[float], out_path: str):
    """
    画 Accuracy 随 epoch 变化曲线并保存
    """
    epochs = list(range(1, len(train_acc) + 1))
    plt.figure()
    plt.plot(epochs, train_acc, label="train_acc")
    plt.plot(epochs, val_acc, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("DKT Choice Prediction Accuracy vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="path to jsonl")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--use_correct", type=int, default=1, choices=[0, 1],
                    help="1: token里包含是否做对；0: 只用选项")
    ap.add_argument("--max_lines", type=int, default=0,
                    help="只读取前N行用于试跑；0表示读取全部")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--emb_dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--q_emb_dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot_path", type=str, default="acc_curve0.png",
                    help="accuracy曲线保存路径")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # 1) 构造样本
    samples, qid2idx = build_samples(
        args.data,
        window=args.window,
        use_correct=args.use_correct,
        max_lines=args.max_lines
    )

    if len(samples) == 0:
        raise RuntimeError(
            "样本为0：可能原因：\n"
            "1) 你只读了很少行(max_lines太小)，导致学生凑不够31条\n"
            "2) 字段名不一致\n"
            "3) Student_Answer 不是A/B/C/D\n"
        )

    Q = len(qid2idx)
    V = Q * (8 if args.use_correct else 4)

    print(f"num_questions={Q}, vocab_size={V}, num_samples={len(samples)}")

    # 2) 严格按学生切分
    train_idx, val_idx, test_idx = split_users(samples, seed=args.seed)
    train_ds = KTWindowDataset([samples[i] for i in train_idx])
    val_ds = KTWindowDataset([samples[i] for i in val_idx])
    test_ds = KTWindowDataset([samples[i] for i in test_idx])

    print(f"train/val/test samples = {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # 3) 模型
    model = DKTChoiceModel(
        vocab_size=V,
        num_questions=Q,
        emb_dim=args.emb_dim,
        lstm_hidden=args.hidden,
        q_emb_dim=args.q_emb_dim
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    # 用来画曲线
    train_acc_curve: List[float] = []
    val_acc_curve: List[float] = []

    # 记录验证集最好的模型
    best_val_acc = -1.0
    best_state = None

    # 4) 训练循环
    for ep in range(1, args.epochs + 1):
        model.train()

        # 统计训练集 loss/acc（acc为了画曲线）
        total_loss = 0.0
        total_n = 0
        total_correct = 0

        for hist_tokens, q_next, a_next in train_loader:
            hist_tokens = hist_tokens.to(device)
            q_next = q_next.to(device)
            a_next = a_next.to(device)

            logits = model(hist_tokens, q_next)   # [B, 4]
            loss = ce(logits, a_next)

            opt.zero_grad()
            loss.backward()
            opt.step()

            # 统计
            total_loss += loss.item() * a_next.size(0)
            total_n += a_next.size(0)
            pred = logits.argmax(dim=-1)
            total_correct += (pred == a_next).sum().item()

        train_ce = total_loss / max(total_n, 1)
        train_acc = total_correct / max(total_n, 1)

        # 验证集
        val_metrics = evaluate(model, val_loader, device)
        val_acc = val_metrics["acc"]

        train_acc_curve.append(train_acc)
        val_acc_curve.append(val_acc)

        print(
            f"Epoch {ep:02d} | "
            f"train_ce={train_ce:.4f} | train_acc={train_acc:.4f} | "
            f"val_acc={val_metrics['acc']:.4f} | val_top2={val_metrics['top2_acc']:.4f} | val_nll={val_metrics['nll']:.4f}"
        )

        # 保存最好验证准确率的模型参数
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # 5) 用最佳模型在 test 上评估
    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, device)
    print("==== Test ====")
    print(test_metrics)

    # 6) 画 accuracy 曲线并保存
    plot_acc_curve(train_acc_curve, val_acc_curve, args.plot_path)
    print(f"[saved] accuracy curve -> {args.plot_path}")


if __name__ == "__main__":
    main()
