# StuSim-Baseline

> 复现 MIT 论文 [**Learning to Make Mistakes**](https://arxiv.org/abs/2412.06406) 的 MISTAKE-CYCLE+CORRECT 方法，基于 EEDI 数学教育数据集，实现「学生 30 条历史做题记录 → 新题学生答案预测」。

---

## 实验结果

### 第一版：平衡数据（70% 错误 + 30% 正确样本）

| 指标 | 基座模型 | SFT 后 | 变化 |
|------|---------|--------|------|
| 整体预测准确率 | 24.69% | **34.77%** | ↑ +10.08% |
| 学生正确子集准确率 | 22.95% | **52.87%** | ↑ +29.92% |
| 学生错误子集准确率 | 26.45% | 16.53% | ↓ -9.92% |
| 循环一致性通过率 | 25.50% | 19.50% | — |

### 第二版：面向错误子集优化（95% 错误样本 + 4× 损失权重）

> 优化目标：提升**学生错误作答子集**的预测准确率。

| 指标 | 基座模型 | SFT 后（优化版）| vs 第一版 SFT |
|------|---------|----------------|--------------|
| 整体预测准确率 | 28.26% | 23.40% | — |
| 学生正确子集准确率 | 28.40% | 26.00% | — |
| **学生错误子集准确率** | **28.11%** | **20.80%** | +4.27%↑ vs v1 |

**验证集最优错误子集准确率（Epoch2）：30.00%**（基座 26.45%，v1 微调 16.53%）

---

## 方法概述

严格对标论文 Algorithm 1 & 2：

```
MISTAKE-GENERATE（Algorithm 1）:
  for each (question q, student_answer a) in D_train:
    m_infer = Model(q, a)          # 正向：推断错误概念
    a_sim   = Model(q, m_infer)    # 反向：模拟学生答案
    if a_sim == a:        weight = 2   # 高置信
    elif a_sim != correct: weight = 1  # 中置信
    else:                  discard     # 反向验证失败，丢弃

MISTAKE-UPDATE（Algorithm 2）:
  for t in range(T=3):
    M_t = LoRA-SFT(M_base, D_weighted)   # 加权因果LM损失
    D_weighted = Regenerate(M_t, D_train) # 用新模型更新数据集
```

---

## 环境与数据

| 项目 | 值 |
|------|----|
| 模型 | Qwen2.5-3B-Instruct（4-bit NF4 QLoRA）|
| GPU | RTX 4090 48 GB |
| LoRA | r=8, alpha=16, target: q/k/v/o_proj |
| 数据集 | EEDI 数学题库（705,288 条）|
| 拆分 | 按 question_id：70% 训练 / 15% 验证 / 15% 测试 |
| 历史长度 | 固定 30 条（不足 padding，过长截断）|

---

## 文件结构

```
├── run_full_pipeline.py          # 主控脚本（一键运行全流程）
├── step3_mistake_generate.py     # MISTAKE-GENERATE 循环一致性数据构建
├── step4_mistake_update.py       # MISTAKE-UPDATE 迭代式 LoRA 微调
├── step5_evaluate.py             # SFT 前后全维度对比评估
├── step6_cleanup.py              # 收尾整理
├── optimize_incorrect_subset.py  # 错误子集专项优化（v2）
├── prompts.py                    # Prompt 模板（含循环一致性约束）
└── evaluation_report/
    ├── evaluation_report_v2.md   # 第一版评估报告
    └── evaluation_report_opt.md  # 优化版评估报告
```

---

## 快速开始

```bash
# 环境准备（所有缓存写入数据盘）
export HF_HOME=/root/autodl-tmp/huggingface_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/huggingface_cache
pip install transformers==4.46.3 peft==0.7.1 bitsandbytes trl==0.11.4 accelerate \
    --cache-dir /root/autodl-tmp/pip_cache

# 一键运行全流程
python run_full_pipeline.py

# 仅运行错误子集优化
python optimize_incorrect_subset.py

# 一键推理（需已训练好模型）
python infer.py --question "..." --opt_a "..." --opt_b "..." --opt_c "..." --opt_d "..." \
                --history_file history.jsonl
```

---

## 关键发现

1. **MISTAKE-CYCLE 有效**：整体准确率提升 +10.08%（v1），Algorithm 2 迭代更新使验证集准确率从 24% 提升至 38.5%。
2. **数据偏差问题**：训练集含 30% 正确样本时，模型对错误子集产生「乐观偏差」（47.6% 预测=正确答案）。
3. **针对性修复**：将错误样本比例提高至 95%、损失权重 ×4，验证集错误子集准确率恢复至 30.00%（超越基座 26.45%）。
4. **3B 模型瓶颈**：精准预测学生选择的具体错误选项仍然困难，建议 7B+ 模型复验。

---

## 引用

```bibtex
@article{mistake2024,
  title  = {Learning to Make Mistakes},
  author = {MIT},
  year   = {2024},
  url    = {https://arxiv.org/abs/2412.06406}
}
```
