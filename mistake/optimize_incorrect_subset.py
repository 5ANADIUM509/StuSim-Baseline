#!/usr/bin/env python3
"""
错误子集预测准确率专项优化脚本

问题根因:
  - SFT数据30%为正确样本 → 模型偏向预测"学生答对"
  - 微调模型在错误子集上47.6%预测=正确答案
  
核心修复:
  1. SFT数据重构: 95%错误样本 + 5%正确样本
  2. 损失权重强化: 错误样本权重从alpha升至alpha×4
  3. 新Prompt设计: 强制模型识别"具体错误选项"
  4. 从基座重新训练(消除旧模型偏差)
"""

import os, sys, json, random, logging, time, re, shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter, defaultdict
from datetime import datetime
from dataclasses import dataclass

# ========== 强制数据盘路径 ==========
os.environ.update({
    "HF_HOME": "/root/autodl-tmp/huggingface_cache",
    "TRANSFORMERS_CACHE": "/root/autodl-tmp/huggingface_cache",
    "TORCH_HOME": "/root/autodl-tmp/torch_cache",
    "TRANSFORMERS_OFFLINE": "1",
})

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel

# ========== 路径配置 ==========
BASE_DIR = Path("/root/autodl-tmp")
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
SFT_OPT_DIR = BASE_DIR / "sft_dataset_opt"
TRAINED_OPT_DIR = BASE_DIR / "trained_model_opt"
EVAL_OPT_DIR = BASE_DIR / "evaluation_report"
LOG_DIR = BASE_DIR / "logs"
TEMP_DIR = BASE_DIR / "temp"
MODEL_PATH = BASE_DIR / "models/Qwen/Qwen2.5-3B-Instruct"

for d in [SFT_OPT_DIR, TRAINED_OPT_DIR, EVAL_OPT_DIR, LOG_DIR, TEMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ========== 日志 ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "optimize_incorrect_subset.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
random.seed(42)
torch.manual_seed(42)

# ========== 超参数 ==========
PIPELINE_START = time.time()
TARGET_SFT_SIZE = 6000          # 训练集大小（全错误样本）
INCORRECT_RATIO = 0.95          # 错误样本占比
INCORRECT_LOSS_MULTIPLIER = 4.0 # 错误样本损失倍率
LORA_R = 8
LORA_ALPHA = 16
BATCH_SIZE = 1
GRAD_ACCUM = 16                 # 有效批大小=16
MAX_EPOCHS = 3
LR = 2e-4
MAX_SEQ_LEN = 2048
TRAIN_SUBSET = 2000             # 每轮训练最高权重的N条
ITERATIONS = 3

# ==================== 新Prompt：强制识别具体错误选项 ====================
ERROR_PREDICTION_SYSTEM = """You are an expert mathematics educator specialized in predicting student errors.

Your PRIMARY task is to predict WHICH SPECIFIC WRONG ANSWER a student will choose, based on their error patterns.

CRITICAL RULES:
1. Analyze the student's MISTAKE PATTERNS from history - which wrong options do they repeatedly choose?
2. For each wrong option in the new problem, identify the SPECIFIC misconception that would cause it
3. Match the student's HISTORICAL ERROR PATTERNS to these misconceptions
4. If the student has > 30% error rate in history, they LIKELY make errors on the new problem too
5. The predicted_student_option should reflect their ACTUAL behavior, not what they should do

CYCLE-CONSISTENCY CONSTRAINT: The inferred misconception must be RESTORABLE - 
applying that misconception when solving must yield the student's predicted choice."""

ERROR_PREDICTION_USER = """Predict this student's answer based on their 30-question error history.

=== STUDENT HISTORY ===
{history_text}

=== NEW QUESTION ===
{new_question}
A. {new_opt_a}
B. {new_opt_b}
C. {new_opt_c}
D. {new_opt_d}

=== ANALYSIS INSTRUCTIONS ===
Step 1 - QUANTIFY ERROR PATTERNS:
Count how many times the student answered incorrectly in their history.
Identify which TYPES of mistakes they make (e.g., sign errors, formula confusion, misreading).

Step 2 - WRONG OPTION ANALYSIS:
For each option A/B/C/D that is NOT correct, describe EXACTLY what misconception leads there.

Step 3 - PATTERN MATCHING:
Does the student's dominant error pattern MATCH any of the wrong options?
If yes → predict that wrong option (student is likely to repeat their mistake).
If no match → predict based on overall error frequency.

Step 4 - OUTPUT:
- predicted_student_option: The option this specific student will MOST LIKELY choose (may be wrong)
- predicted_correct_option: The mathematically correct option

Output ONLY valid JSON, no extra text:
{{"error_rate": "X/30 incorrect", "dominant_error_type": "...", "matched_wrong_option": "X or None", "predicted_student_option": "X", "predicted_correct_option": "Y"}}"""

# 循环一致性校验的短Prompt（保持不变）
CYCLE_FORWARD_PROMPT = """Question: {question}
Options: A) {opt_a} B) {opt_b} C) {opt_c} D) {opt_d}
Student chose: {student_ans} (Correct: {correct_ans})

What specific mathematical misconception caused the student to choose {student_ans}?
Give ONE sentence, be specific about what they confused or misapplied.

Misconception:"""

CYCLE_BACKWARD_PROMPT = """Question: {question}
Options: A) {opt_a}  B) {opt_b}  C) {opt_c}  D) {opt_d}

A student applying this misconception: "{misconception}"

Which option would they choose? Reply ONLY with A, B, C, or D.

Answer:"""


def build_error_focused_prompt(sample: Dict) -> str:
    """构建强化错误预测的Prompt"""
    user_content = ERROR_PREDICTION_USER.format(
        history_text=sample["history_text"],
        new_question=sample["new_question"],
        new_opt_a=sample["new_opt_a"],
        new_opt_b=sample["new_opt_b"],
        new_opt_c=sample["new_opt_c"],
        new_opt_d=sample["new_opt_d"],
    )
    return (
        f"<|im_start|>system\n{ERROR_PREDICTION_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_error_response(sample: Dict, misconception: str) -> str:
    """构建错误预测的目标响应（格式与新Prompt匹配）"""
    # 计算历史错误率
    history_text = sample.get("history_text", "")
    error_marks = history_text.count("✗")
    total_marks = history_text.count("✓") + error_marks
    error_rate_str = f"{error_marks}/{total_marks} incorrect" if total_marks > 0 else "unknown"
    
    student_ans = sample["student_answer"]
    correct_ans = sample["correct_answer"]
    is_correct = sample.get("is_correct", student_ans == correct_ans)
    
    if is_correct:
        return json.dumps({
            "error_rate": error_rate_str,
            "dominant_error_type": "Student demonstrates mastery - no dominant error pattern",
            "matched_wrong_option": "None",
            "predicted_student_option": student_ans,
            "predicted_correct_option": correct_ans,
        })
    else:
        return json.dumps({
            "error_rate": error_rate_str,
            "dominant_error_type": misconception,
            "matched_wrong_option": student_ans,
            "predicted_student_option": student_ans,
            "predicted_correct_option": correct_ans,
        })


# ==================== 循环一致性校验 ====================
def load_model_4bit(model_path: str, lora_path: Optional[str] = None):
    logger.info(f"加载模型: {model_path}" + (f" + LoRA: {lora_path}" if lora_path else ""))
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb_config, device_map="auto", trust_remote_code=True,
    )
    if lora_path and Path(lora_path).exists():
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return tokenizer, model


def batch_generate(tokenizer, model, prompts: List[str], max_new_tokens=60, batch_size=16, temperature=0.3) -> List[str]:
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i: i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0), temperature=temperature if temperature > 0 else 1.0,
                top_p=0.9, pad_token_id=tokenizer.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        for output in outputs:
            results.append(tokenizer.decode(output[input_len:], skip_special_tokens=True).strip())
        if (i // batch_size + 1) % 20 == 0:
            logger.info(f"  推理进度: {min(i+batch_size, len(prompts))}/{len(prompts)}")
    return results


def extract_misconception(text: str) -> str:
    text = text.strip()
    for prefix in ["Misconception:", "The misconception", "The student"]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].lstrip(": ").strip()
    return text.split(".")[0].strip()[:200] if text else "Incorrect rule application"


def parse_option(text: str) -> Optional[str]:
    m = re.search(r'\b([ABCD])\b', text.strip().upper())
    return m.group(1) if m else None


def run_cycle_consistency(tokenizer, model, incorrect_samples: List[Dict]) -> List[Dict]:
    """对错误样本执行循环一致性校验"""
    logger.info(f"循环一致性校验: {len(incorrect_samples)} 个错误样本")
    
    # 正向推断
    fwd_prompts = [
        CYCLE_FORWARD_PROMPT.format(
            question=s["new_question"],
            opt_a=s["new_opt_a"], opt_b=s["new_opt_b"],
            opt_c=s["new_opt_c"], opt_d=s["new_opt_d"],
            student_ans=s["student_answer"], correct_ans=s["correct_answer"],
        ) for s in incorrect_samples
    ]
    logger.info("正向推断错误概念...")
    fwd_outputs = batch_generate(tokenizer, model, fwd_prompts, max_new_tokens=60, batch_size=16, temperature=0.3)
    misconceptions = [extract_misconception(o) for o in fwd_outputs]
    
    # 反向模拟
    bwd_prompts = [
        CYCLE_BACKWARD_PROMPT.format(
            question=s["new_question"],
            opt_a=s["new_opt_a"], opt_b=s["new_opt_b"],
            opt_c=s["new_opt_c"], opt_d=s["new_opt_d"],
            misconception=m,
        ) for s, m in zip(incorrect_samples, misconceptions)
    ]
    logger.info("反向模拟答案...")
    bwd_outputs = batch_generate(tokenizer, model, bwd_prompts, max_new_tokens=5, batch_size=32, temperature=0.0)
    sim_answers = [parse_option(o) for o in bwd_outputs]
    
    results = []
    stats = Counter()
    for s, m, a_sim in zip(incorrect_samples, misconceptions, sim_answers):
        if a_sim == s["student_answer"]:
            weight = 2
            wt = "high"
        elif a_sim == s["correct_answer"]:
            weight = 0  # 丢弃：反向验证失败，推出正确答案
            wt = "discard"
        else:
            weight = 1
            wt = "medium"
        stats[wt] += 1
        r = dict(s)
        r.update({"misconception": m, "simulated_answer": a_sim, "weight": weight, "weight_type": wt})
        results.append(r)
    
    logger.info(f"权重分布: {dict(stats)}")
    logger.info(f"有效样本: {sum(1 for r in results if r['weight']>0)}")
    return results


def build_optimized_sft_dataset(
    incorrect_samples_with_weights: List[Dict],
    correct_samples: List[Dict],
    target_size: int = TARGET_SFT_SIZE,
) -> List[Dict]:
    """
    构建优化版SFT数据集
    
    关键改变:
    - 错误样本: 95% (强化错误预测信号)
    - 正确样本: 5% (仅保留少量用于平衡)
    - 错误样本损失权重: 原alpha × INCORRECT_LOSS_MULTIPLIER
    """
    valid_incorrect = [s for s in incorrect_samples_with_weights if s["weight"] > 0]
    valid_incorrect.sort(key=lambda x: x["weight"], reverse=True)
    
    n_incorrect = int(target_size * INCORRECT_RATIO)
    n_correct = target_size - n_incorrect
    
    selected_incorrect = valid_incorrect[:n_incorrect]
    selected_correct = random.sample(correct_samples, min(n_correct, len(correct_samples)))
    
    logger.info(f"优化SFT数据: 错误={len(selected_incorrect)}, 正确={len(selected_correct)}")
    
    sft_data = []
    
    for s in selected_incorrect:
        prompt = build_error_focused_prompt(s)
        response = build_error_response(s, s.get("misconception", "Incorrect rule application"))
        # 关键: 错误样本损失权重 = 原alpha × 倍率（最大化错误预测信号）
        effective_weight = s["weight"] * INCORRECT_LOSS_MULTIPLIER
        sft_data.append({
            "prompt": prompt,
            "response": response,
            "weight": effective_weight,
            "base_weight": s["weight"],
            "weight_type": s.get("weight_type", "medium") + "_error_boosted",
            "student_answer": s["student_answer"],
            "correct_answer": s["correct_answer"],
            "simulated_answer": s.get("simulated_answer"),
            "misconception": s.get("misconception", ""),
            "is_correct": False,
            "new_question": s.get("new_question", ""),
            "new_opt_a": s.get("new_opt_a", ""),
            "new_opt_b": s.get("new_opt_b", ""),
            "new_opt_c": s.get("new_opt_c", ""),
            "new_opt_d": s.get("new_opt_d", ""),
            "history_text": s.get("history_text", ""),
        })
    
    for s in selected_correct:
        prompt = build_error_focused_prompt(s)
        response = build_error_response(s, "No dominant error pattern - student demonstrates mastery")
        sft_data.append({
            "prompt": prompt,
            "response": response,
            "weight": 1.0,  # 正确样本保持低权重
            "base_weight": 1,
            "weight_type": "correct_low_weight",
            "student_answer": s["student_answer"],
            "correct_answer": s["correct_answer"],
            "simulated_answer": s["student_answer"],
            "misconception": "",
            "is_correct": True,
            "new_question": s.get("new_question", ""),
            "new_opt_a": s.get("new_opt_a", ""),
            "new_opt_b": s.get("new_opt_b", ""),
            "new_opt_c": s.get("new_opt_c", ""),
            "new_opt_d": s.get("new_opt_d", ""),
            "history_text": s.get("history_text", ""),
        })
    
    sft_data.sort(key=lambda x: x["weight"], reverse=True)
    logger.info(f"优化SFT数据集总计: {len(sft_data)} 条")
    
    wt_dist = Counter(round(d["weight"]) for d in sft_data)
    logger.info(f"有效权重分布: {dict(wt_dist)}")
    
    return sft_data


# ==================== 优化版SFT Dataset ====================
class OptimizedSFTDataset(Dataset):
    def __init__(self, samples: List[Dict], tokenizer, max_seq_len: int = MAX_SEQ_LEN):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        prompt, response, weight = s["prompt"], s["response"], float(s["weight"])
        
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        response_ids = self.tokenizer(
            response + self.tokenizer.eos_token, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0]
        
        input_ids = torch.cat([prompt_ids, response_ids])
        labels = torch.cat([torch.full_like(prompt_ids, -100), response_ids])
        
        if len(input_ids) > self.max_seq_len:
            resp_len = len(response_ids)
            max_prompt = self.max_seq_len - resp_len
            if max_prompt < 50:
                max_prompt = 50
                response_ids = response_ids[:self.max_seq_len - 50]
            prompt_ids = prompt_ids[-max_prompt:]
            input_ids = torch.cat([prompt_ids, response_ids])
            labels = torch.cat([torch.full((len(prompt_ids),), -100, dtype=torch.long), response_ids])
        
        return {"input_ids": input_ids, "labels": labels, "weight": torch.tensor(weight, dtype=torch.float32)}


def collate_fn(batch):
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids_list, labels_list, attn_mask_list, weights_list = [], [], [], []
    for b in batch:
        seq_len = b["input_ids"].shape[0]
        pad_len = max_len - seq_len
        input_ids_list.append(torch.cat([torch.zeros(pad_len, dtype=torch.long), b["input_ids"]]))
        labels_list.append(torch.cat([torch.full((pad_len,), -100, dtype=torch.long), b["labels"]]))
        attn_mask_list.append(torch.cat([torch.zeros(pad_len, dtype=torch.long), torch.ones(seq_len, dtype=torch.long)]))
        weights_list.append(b["weight"])
    return {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
        "attention_mask": torch.stack(attn_mask_list),
        "weights": torch.stack(weights_list),
    }


def compute_weighted_loss(logits, labels, weights):
    bs, sl, vs = logits.shape
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    per_token_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
        shift_logits.view(-1, vs), shift_labels.view(-1)
    ).view(bs, -1)
    mask = (shift_labels != -100).float()
    per_sample_loss = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    weights = weights.to(per_sample_loss.device)
    weights = weights / weights.mean()
    return (per_sample_loss * weights).mean()


# ==================== 评估（专注错误子集）====================
def evaluate_on_val(model, tokenizer, val_samples: List[Dict]) -> Dict[str, float]:
    """在验证集上评估，分别统计正确和错误子集准确率"""
    model.eval()
    device = next(model.parameters()).device
    
    correct_hits, correct_total = 0, 0
    incorrect_hits, incorrect_total = 0, 0
    
    batch_size = 4
    samples_to_eval = val_samples[:min(200, len(val_samples))]
    
    for i in range(0, len(samples_to_eval), batch_size):
        batch = samples_to_eval[i: i + batch_size]
        prompts = [build_error_focused_prompt(s) for s in batch]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=100, do_sample=False, temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        
        for j, (s, output) in enumerate(zip(batch, outputs)):
            generated = tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()
            predicted = parse_predicted_option(generated)
            
            if s["is_correct"]:
                correct_total += 1
                if predicted == s["student_answer"]:
                    correct_hits += 1
            else:
                incorrect_total += 1
                if predicted == s["student_answer"]:
                    incorrect_hits += 1
    
    model.train()
    return {
        "overall_acc": (correct_hits + incorrect_hits) / max(correct_total + incorrect_total, 1),
        "correct_acc": correct_hits / max(correct_total, 1),
        "incorrect_acc": incorrect_hits / max(incorrect_total, 1),
        "incorrect_hits": incorrect_hits,
        "incorrect_total": incorrect_total,
    }


def parse_predicted_option(text: str) -> Optional[str]:
    """从模型输出中解析预测的学生选项"""
    try:
        m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if m:
            parsed = json.loads(m.group())
            v = parsed.get("predicted_student_option", "")
            if str(v).upper() in "ABCD" and len(str(v)) == 1:
                return str(v).upper()
    except:
        pass
    m = re.search(r'"predicted_student_option"\s*:\s*"([ABCD])"', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


# ==================== 迭代训练 ====================
def load_model_for_training(model_path: str):
    """加载用于QLoRA训练的模型"""
    logger.info(f"加载训练模型: {model_path}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb_config, device_map="auto", trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, task_type=TaskType.CAUSAL_LM, bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return tokenizer, model


def train_one_epoch(model, tokenizer, train_samples: List[Dict], device: str = "cuda") -> float:
    """训练一个epoch，返回平均loss"""
    dataset = OptimizedSFTDataset(train_samples, tokenizer, MAX_SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(len(dataloader) * 0.05),
        num_training_steps=len(dataloader),
    )
    
    scaler = torch.amp.GradScaler("cuda")
    total_loss = 0.0
    optimizer.zero_grad()
    
    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        weights = batch["weights"]
        
        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=None)
            loss = compute_weighted_loss(outputs.logits, labels, weights)
            loss = loss / GRAD_ACCUM
        
        scaler.scale(loss).backward()
        
        if (step + 1) % GRAD_ACCUM == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * GRAD_ACCUM
        
        if (step + 1) % 100 == 0:
            logger.info(f"  Step {step+1}/{len(dataloader)} Loss={total_loss/(step+1):.4f}")
    
    return total_loss / len(dataloader)


def run_iterative_training(
    tokenizer,
    model,
    sft_data: List[Dict],
    val_samples: List[Dict],
) -> Tuple[float, str]:
    """迭代式训练，以错误子集准确率为最优标准"""
    best_incorrect_acc = 0.0
    best_model_path = str(TRAINED_OPT_DIR / "best_model")
    no_improve = 0
    
    current_data = sft_data
    
    for iteration in range(ITERATIONS):
        logger.info(f"\n{'='*50}\n迭代 {iteration+1}/{ITERATIONS}\n{'='*50}")
        
        # 取最高权重样本
        sorted_data = sorted(current_data, key=lambda x: x["weight"], reverse=True)
        train_subset = sorted_data[:TRAIN_SUBSET]
        
        wt_dist = Counter(round(d["weight"]) for d in train_subset)
        inc_count = sum(1 for d in train_subset if not d.get("is_correct", False))
        logger.info(f"训练集: {len(train_subset)}条, 错误={inc_count}({inc_count/len(train_subset):.1%}), 权重分布: {dict(wt_dist)}")
        
        best_epoch_acc = 0.0
        
        for epoch in range(MAX_EPOCHS):
            epoch_start = time.time()
            avg_loss = train_one_epoch(model, tokenizer, train_subset)
            epoch_time = time.time() - epoch_start
            
            logger.info(f"Iter{iteration+1} Epoch{epoch+1} 完成: Loss={avg_loss:.4f}, 耗时={epoch_time:.0f}s")
            
            # 评估（重点关注错误子集）
            metrics = evaluate_on_val(model, tokenizer, val_samples)
            logger.info(
                f"验证集: 整体={metrics['overall_acc']:.4f}, "
                f"正确={metrics['correct_acc']:.4f}, "
                f"错误={metrics['incorrect_acc']:.4f} ({metrics['incorrect_hits']}/{metrics['incorrect_total']})"
            )
            
            # 以错误子集准确率为最优标准（核心改变）
            if metrics["incorrect_acc"] > best_incorrect_acc:
                best_incorrect_acc = metrics["incorrect_acc"]
                best_epoch_acc = metrics["incorrect_acc"]
                no_improve = 0
                
                os.makedirs(best_model_path, exist_ok=True)
                model.save_pretrained(best_model_path)
                tokenizer.save_pretrained(best_model_path)
                
                with open(Path(best_model_path) / "training_info.json", "w") as f:
                    json.dump({
                        "iteration": iteration + 1, "epoch": epoch + 1,
                        "best_incorrect_acc": best_incorrect_acc,
                        **metrics
                    }, f, indent=2)
                
                logger.info(f"✅ 保存最优模型 (incorrect_acc={best_incorrect_acc:.4f})")
            else:
                no_improve += 1
                logger.info(f"错误子集无提升，连续{no_improve}轮")
            
            if no_improve >= 1:
                logger.info("早停触发")
                break
        
        # 检查时间
        if time.time() - PIPELINE_START > 6000:  # 100分钟
            logger.warning("接近时间限制，提前结束训练")
            break
        
        # 迭代更新数据集（反向校验过滤低质量样本）
        if iteration < ITERATIONS - 1:
            logger.info("更新数据集（用当前模型重新校验）...")
            current_data = refresh_sft_data(model, tokenizer, current_data)
    
    return best_incorrect_acc, best_model_path


def refresh_sft_data(model, tokenizer, sft_data: List[Dict]) -> List[Dict]:
    """用当前模型重新校验错误样本，更新权重"""
    model.eval()
    
    incorrect_samples = [s for s in sft_data if not s.get("is_correct", False)]
    
    # 反向校验：确认misconception能还原student_answer
    bwd_prompts = [
        CYCLE_BACKWARD_PROMPT.format(
            question=s.get("new_question", ""),
            opt_a=s.get("new_opt_a", ""), opt_b=s.get("new_opt_b", ""),
            opt_c=s.get("new_opt_c", ""), opt_d=s.get("new_opt_d", ""),
            misconception=s.get("misconception", ""),
        ) for s in incorrect_samples[:2000]
    ]
    
    bwd_outputs = batch_generate(tokenizer, model, bwd_prompts, max_new_tokens=5, batch_size=32, temperature=0.0)
    sim_answers = [parse_option(o) for o in bwd_outputs]
    
    updated = []
    stats = Counter()
    for i, s in enumerate(sft_data):
        if s.get("is_correct", False) or i >= len(incorrect_samples) or i >= len(sim_answers):
            updated.append(s)
            continue
        
        sim_ans = sim_answers[i]
        new_s = dict(s)
        base_w = s.get("base_weight", 1)
        
        if sim_ans == s["student_answer"]:
            new_s["weight"] = base_w * INCORRECT_LOSS_MULTIPLIER
            new_s["simulated_answer_refresh"] = sim_ans
            stats["verified"] += 1
        elif sim_ans == s["correct_answer"]:
            new_s["weight"] = 0  # 丢弃
            stats["discarded"] += 1
        else:
            new_s["weight"] = (base_w / 2) * INCORRECT_LOSS_MULTIPLIER
            stats["partial"] += 1
        
        updated.append(new_s)
    
    valid = [s for s in updated if s.get("weight", 0) > 0]
    valid.sort(key=lambda x: x["weight"], reverse=True)
    logger.info(f"数据集更新: {dict(stats)}, 有效样本={len(valid)}")
    
    model.train()
    return valid


# ==================== 全量评估 ====================
def evaluate_all_metrics(
    base_tokenizer, base_model,
    ft_tokenizer, ft_model,
    test_samples: List[Dict],
    n_eval: int = 500,
) -> Dict:
    """对基座和微调模型执行全量评估"""
    
    # 均衡采样
    correct_test = [s for s in test_samples if s["is_correct"]]
    incorrect_test = [s for s in test_samples if not s["is_correct"]]
    
    n_each = n_eval // 2
    eval_set = (
        random.sample(correct_test, min(n_each, len(correct_test))) +
        random.sample(incorrect_test, min(n_each, len(incorrect_test)))
    )
    
    def run_eval(tokenizer, model, samples, label):
        model.eval()
        device = next(model.parameters()).device
        results = []
        
        for i in range(0, len(samples), 4):
            batch = samples[i: i + 4]
            prompts = [build_error_focused_prompt(s) for s in batch]
            
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=120, do_sample=False, temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )
            input_len = inputs["input_ids"].shape[1]
            
            for s, output in zip(batch, outputs):
                generated = tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()
                predicted = parse_predicted_option(generated)
                parse_ok = predicted is not None
                
                results.append({
                    "user_id": s["user_id"],
                    "question_id": s["question_id"],
                    "student_answer": s["student_answer"],
                    "correct_answer": s["correct_answer"],
                    "is_correct": s["is_correct"],
                    "predicted": predicted,
                    "parse_success": parse_ok,
                    "raw": generated[:200],
                })
        
        # 计算指标（只用parse成功的）
        valid = [r for r in results if r["parse_success"]]
        correct_sub = [r for r in valid if r["is_correct"]]
        incorrect_sub = [r for r in valid if not r["is_correct"]]
        
        def acc(sub):
            if not sub: return 0.0, 0, 0
            h = sum(1 for r in sub if r["predicted"] == r["student_answer"])
            return h/len(sub), h, len(sub)
        
        o_acc, o_h, o_t = acc(valid)
        c_acc, c_h, c_t = acc(correct_sub)
        i_acc, i_h, i_t = acc(incorrect_sub)
        # 正确答案预测准确率
        ca_acc = sum(1 for r in valid if r["predicted"] == r["correct_answer"]) / max(len(valid), 1)
        
        logger.info(f"\n[{label}] 评估结果 ({len(valid)}/{len(results)} 解析成功):")
        logger.info(f"  整体: {o_acc:.4f} ({o_h}/{o_t})")
        logger.info(f"  正确子集: {c_acc:.4f} ({c_h}/{c_t})")
        logger.info(f"  错误子集: {i_acc:.4f} ({i_h}/{i_t}) ← 核心指标")
        
        return {
            "label": label, "total": len(results), "valid": len(valid),
            "overall_accuracy": o_acc, "overall_hits": o_h, "overall_total": o_t,
            "correct_subset_accuracy": c_acc, "correct_subset_hits": c_h, "correct_subset_total": c_t,
            "incorrect_subset_accuracy": i_acc, "incorrect_subset_hits": i_h, "incorrect_subset_total": i_t,
            "correct_answer_accuracy": ca_acc,
        }
    
    logger.info("\n评估SFT前基座模型...")
    base_metrics = run_eval(base_tokenizer, base_model, eval_set, "SFT前基座")
    
    del base_model
    torch.cuda.empty_cache()
    
    logger.info("\n评估SFT后微调模型（优化版）...")
    ft_metrics = run_eval(ft_tokenizer, ft_model, eval_set, "SFT后优化版")
    
    del ft_model
    torch.cuda.empty_cache()
    
    return base_metrics, ft_metrics, eval_set


def generate_comparison_report(
    base_metrics: Dict, ft_metrics: Dict,
    prev_ft_metrics: Dict,  # 上一版本微调结果用于对比
    output_path: str,
):
    """生成三方对比报告（基座 vs 旧版微调 vs 新版优化）"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def pct(v): return f"{v*100:.2f}%" if v is not None else "N/A"
    def delta(v1, v2):
        if v1 is None or v2 is None: return "N/A"
        d = (v2 - v1) * 100
        return f"{'↑' if d>0 else '↓'} {abs(d):.2f}%"
    
    report = f"""# 错误子集专项优化 - 效果对比报告

> 生成时间: {now}  
> 优化目标: 提升学生错误作答子集预测准确率  

---

## 核心改进摘要

| 改进项 | 旧版 | 优化版 |
|--------|------|--------|
| 错误样本比例 | 70% | 95% |
| 错误样本损失权重 | α×1 | α×{INCORRECT_LOSS_MULTIPLIER} |
| Prompt设计 | 通用预测 | **强制识别具体错误选项** |
| 最优标准 | 整体准确率 | **错误子集准确率** |
| 正确答案偏差 | 47.6% | 待测量 |

---

## 三方对比：基座 vs 旧版微调 vs 新版优化

### 整体预测准确率

| 模型 | 准确率 | vs 基座 |
|------|--------|---------|
| SFT前（基座） | {pct(base_metrics.get('overall_accuracy'))} | — |
| SFT后（旧版微调） | {pct(prev_ft_metrics.get('overall_accuracy'))} | {delta(base_metrics.get('overall_accuracy'), prev_ft_metrics.get('overall_accuracy'))} |
| SFT后（优化版） | {pct(ft_metrics.get('overall_accuracy'))} | {delta(base_metrics.get('overall_accuracy'), ft_metrics.get('overall_accuracy'))} |

### 🎯 学生正确作答子集准确率

| 模型 | 准确率 | vs 基座 |
|------|--------|---------|
| SFT前（基座） | {pct(base_metrics.get('correct_subset_accuracy'))} | — |
| SFT后（旧版微调） | {pct(prev_ft_metrics.get('correct_subset_accuracy'))} | {delta(base_metrics.get('correct_subset_accuracy'), prev_ft_metrics.get('correct_subset_accuracy'))} |
| SFT后（优化版） | {pct(ft_metrics.get('correct_subset_accuracy'))} | {delta(base_metrics.get('correct_subset_accuracy'), ft_metrics.get('correct_subset_accuracy'))} |

### 🎯 学生错误作答子集准确率（核心优化目标）

| 模型 | 准确率 | vs 基座 | vs 旧版 |
|------|--------|---------|---------|
| SFT前（基座） | {pct(base_metrics.get('incorrect_subset_accuracy'))} | — | — |
| SFT后（旧版微调） | {pct(prev_ft_metrics.get('incorrect_subset_accuracy'))} | {delta(base_metrics.get('incorrect_subset_accuracy'), prev_ft_metrics.get('incorrect_subset_accuracy'))} | — |
| **SFT后（优化版）** | **{pct(ft_metrics.get('incorrect_subset_accuracy'))}** | **{delta(base_metrics.get('incorrect_subset_accuracy'), ft_metrics.get('incorrect_subset_accuracy'))}** | **{delta(prev_ft_metrics.get('incorrect_subset_accuracy'), ft_metrics.get('incorrect_subset_accuracy'))}** |

---

## 正确答案偏差分析

| 指标 | 旧版微调 | 优化版 |
|------|---------|--------|
| 错误子集中预测=正确答案比例 | 47.6%（已知） | 待统计 |
| 正确答案预测准确率 | {pct(prev_ft_metrics.get('correct_answer_prediction_accuracy', None))} | {pct(ft_metrics.get('correct_answer_accuracy'))} |

---

## 优化原理

**问题根因**: 旧版微调模型对错误子集样本有47.6%的预测等于正确答案，而非学生实际选择的错误选项。  
这是因为训练数据中30%为正确样本（监督模型"什么时候答对"），导致模型产生"乐观偏差"。

**优化策略**:
1. **数据重构**: 错误样本比例从70%→95%，消除"答对偏差"
2. **损失权重**: 错误样本有效权重×{INCORRECT_LOSS_MULTIPLIER}，放大错误预测的训练信号  
3. **新Prompt**: 引导模型分析错误率→识别具体错误选项→匹配历史错误模式
4. **最优标准**: 以错误子集准确率（而非整体准确率）为模型选择依据

---

*报告由 optimize_incorrect_subset.py 自动生成*
"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"优化对比报告已保存: {output_path}")


# ==================== 主函数 ====================
def main():
    logger.info("=" * 60)
    logger.info("错误子集预测准确率专项优化")
    logger.info("=" * 60)
    
    # ===== 1. 加载数据 =====
    logger.info("加载训练数据...")
    train_samples = []
    with open(PROCESSED_DATA_DIR / "train.jsonl") as f:
        for line in f:
            train_samples.append(json.loads(line.strip()))
    
    incorrect_train = [s for s in train_samples if not s["is_correct"]]
    correct_train = [s for s in train_samples if s["is_correct"]]
    logger.info(f"训练集: 错误={len(incorrect_train)}, 正确={len(correct_train)}")
    
    val_samples = []
    with open(PROCESSED_DATA_DIR / "val.jsonl") as f:
        for line in f:
            val_samples.append(json.loads(line.strip()))
    
    # 均衡采样验证集（多一点错误样本）
    val_correct = [s for s in val_samples if s["is_correct"]]
    val_incorrect = [s for s in val_samples if not s["is_correct"]]
    val_eval = (
        random.sample(val_correct, min(60, len(val_correct))) +
        random.sample(val_incorrect, min(140, len(val_incorrect)))
    )
    logger.info(f"验证集: 正确={len([s for s in val_eval if s['is_correct']])}, 错误={len([s for s in val_eval if not s['is_correct']])}")
    
    # ===== 2. 检查现有SFT数据是否可复用 =====
    opt_sft_path = SFT_OPT_DIR / "sft_opt.jsonl"
    
    if opt_sft_path.exists():
        logger.info(f"✅ 优化SFT数据已存在，直接复用: {opt_sft_path}")
        with open(opt_sft_path) as f:
            sft_data = [json.loads(l) for l in f]
        logger.info(f"加载 {len(sft_data)} 条SFT数据")
    else:
        # ===== 3. 加载模型执行循环一致性校验 =====
        logger.info("加载4-bit基座模型执行循环一致性校验...")
        tokenizer, model = load_model_4bit(str(MODEL_PATH))
        
        # 采样候选集
        n_candidate = int(TARGET_SFT_SIZE * INCORRECT_RATIO * 2.5)
        n_candidate = min(n_candidate, len(incorrect_train))
        candidate_incorrect = random.sample(incorrect_train, n_candidate)
        logger.info(f"循环一致性校验候选: {n_candidate} 条")
        
        # 执行校验
        incorrect_checked = run_cycle_consistency(tokenizer, model, candidate_incorrect)
        
        # 释放推理模型
        del model
        torch.cuda.empty_cache()
        import gc; gc.collect()
        logger.info("推理模型已释放")
        
        # 构建优化SFT数据集
        sft_data = build_optimized_sft_dataset(incorrect_checked, correct_train, TARGET_SFT_SIZE)
        
        # 保存
        with open(opt_sft_path, "w") as f:
            for d in sft_data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        logger.info(f"优化SFT数据已保存: {opt_sft_path}")
    
    # ===== 4. 迭代训练 =====
    logger.info("\n加载训练模型...")
    tokenizer_train, model_train = load_model_for_training(str(MODEL_PATH))
    
    best_acc, best_model_path = run_iterative_training(tokenizer_train, model_train, sft_data, val_eval)
    logger.info(f"\n训练完成! 最优错误子集准确率: {best_acc:.4f}")
    logger.info(f"最优模型: {best_model_path}")
    
    del model_train
    torch.cuda.empty_cache()
    import gc; gc.collect()
    
    # ===== 5. 全量评估 =====
    logger.info("\n加载测试数据...")
    test_samples = []
    with open(PROCESSED_DATA_DIR / "test.jsonl") as f:
        for line in f:
            test_samples.append(json.loads(line.strip()))
    
    logger.info("加载基座模型...")
    base_tok, base_model = load_model_4bit(str(MODEL_PATH))
    
    logger.info("加载优化微调模型...")
    ft_tok, ft_model = load_model_4bit(str(MODEL_PATH), lora_path=best_model_path)
    
    base_m, ft_m, _ = evaluate_all_metrics(base_tok, base_model, ft_tok, ft_model, test_samples, n_eval=500)
    
    # 加载上一版本微调指标用于对比
    prev_metrics_path = BASE_DIR / "evaluation_report/metrics_v2.json"
    prev_ft_metrics = {}
    if prev_metrics_path.exists():
        with open(prev_metrics_path) as f:
            prev_data = json.load(f)
        prev_ft_metrics = {
            "overall_accuracy": prev_data["fine_tuned_model"]["overall_accuracy"],
            "correct_subset_accuracy": prev_data["fine_tuned_model"]["correct_subset_accuracy"],
            "incorrect_subset_accuracy": prev_data["fine_tuned_model"]["incorrect_subset_accuracy"],
            "correct_answer_prediction_accuracy": prev_data["fine_tuned_model"].get("correct_answer_prediction_accuracy"),
        }
    
    # 保存指标
    metrics_output = {
        "base_model": base_m,
        "ft_model_optimized": ft_m,
        "ft_model_previous": prev_ft_metrics,
        "optimization_config": {
            "incorrect_ratio": INCORRECT_RATIO,
            "incorrect_loss_multiplier": INCORRECT_LOSS_MULTIPLIER,
            "train_subset": TRAIN_SUBSET,
            "iterations": ITERATIONS,
        },
        "timestamp": datetime.now().isoformat(),
    }
    with open(EVAL_OPT_DIR / "metrics_opt.json", "w") as f:
        json.dump(metrics_output, f, indent=2, ensure_ascii=False)
    
    # 生成报告
    generate_comparison_report(
        base_m, ft_m, prev_ft_metrics,
        str(EVAL_OPT_DIR / "evaluation_report_opt.md"),
    )
    
    # 打印摘要
    total_time = time.time() - PIPELINE_START
    logger.info("\n" + "=" * 60)
    logger.info("优化结果摘要")
    logger.info("=" * 60)
    logger.info(f"总耗时: {total_time/60:.1f} 分钟")
    logger.info(f"\n错误子集准确率对比:")
    logger.info(f"  基座模型:     {base_m.get('incorrect_subset_accuracy', 0):.2%}")
    logger.info(f"  旧版微调:     {prev_ft_metrics.get('incorrect_subset_accuracy', 0):.2%}")
    logger.info(f"  优化版微调:   {ft_m.get('incorrect_subset_accuracy', 0):.2%}")
    
    improvement = ft_m.get('incorrect_subset_accuracy', 0) - base_m.get('incorrect_subset_accuracy', 0)
    logger.info(f"\n  vs基座提升: {improvement:+.2%}")
    
    vs_prev = ft_m.get('incorrect_subset_accuracy', 0) - prev_ft_metrics.get('incorrect_subset_accuracy', 0)
    logger.info(f"  vs旧版提升: {vs_prev:+.2%}")


if __name__ == "__main__":
    main()
