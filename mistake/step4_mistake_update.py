#!/usr/bin/env python3
"""
步骤4: MISTAKE-UPDATE - 迭代式LoRA微调
严格对标论文Algorithm2，实现加权因果语言建模损失

核心特性:
- LoRA参数: r=8, lora_alpha=16, target_modules=[q/k/v/o_proj]
- 加权损失: 仅对Response部分计算损失，按循环一致性权重加权
- 迭代训练: T=3轮，每轮最多3 epoch，早停策略
- 验证集早停: 连续1轮不提升则停止
"""

import os, sys, json, random, logging, time, shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

# ========== 强制数据盘路径 ==========
os.environ["HF_HOME"] = "/root/autodl-tmp/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "/root/autodl-tmp/huggingface_cache"
os.environ["TORCH_HOME"] = "/root/autodl-tmp/torch_cache"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import (
    LoraConfig, TaskType, get_peft_model,
    prepare_model_for_kbit_training,
)

# ========== 路径配置 ==========
BASE_DIR = Path("/root/autodl-tmp")
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
SFT_DATASET_DIR = BASE_DIR / "sft_dataset"
TRAINED_MODEL_DIR = BASE_DIR / "trained_model"
TEMP_DIR = BASE_DIR / "temp"
LOG_DIR = BASE_DIR / "logs"
MODEL_PATH = BASE_DIR / "models/Qwen/Qwen2.5-3B-Instruct"

TRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ========== 日志配置 ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "step4_mistake_update.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ========== 随机种子 ==========
random.seed(42)
torch.manual_seed(42)

# ========== 训练超参数 ==========
@dataclass
class TrainingConfig:
    max_iterations: int = 3       # 迭代轮次T=3
    max_epochs_per_iter: int = 2  # 每轮最多2 epoch（时间约束）
    early_stop_patience: int = 1  # 连续1轮不提升则停止
    learning_rate: float = 2e-4
    batch_size: int = 1           # 减小以适应2048 token
    gradient_accumulation_steps: int = 16  # 有效批大小=16
    max_seq_len: int = 2048       # 支持完整prompt（~1600 tokens）
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")
    train_top_n: int = 2000       # 每轮取权重最高的N条训练
    val_eval_samples: int = 200   # 验证集评估样本数
    fp16: bool = True


# ========== 数据集类 ==========
class SFTDataset(Dataset):
    """SFT训练数据集，支持加权损失"""
    
    def __init__(
        self,
        samples: List[Dict],
        tokenizer,
        max_seq_len: int = 1024,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt = sample["prompt"]
        response = sample["response"]
        weight = float(sample.get("weight", 1.0))
        
        # 编码prompt（不计算损失的部分）
        prompt_ids = self.tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0]
        
        # 编码response（计算损失的部分）
        response_ids = self.tokenizer(
            response + self.tokenizer.eos_token,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]
        
        # 拼接完整序列
        input_ids = torch.cat([prompt_ids, response_ids])
        
        # 创建标签：prompt部分设为-100（不计算损失）
        labels = torch.cat([
            torch.full_like(prompt_ids, -100),
            response_ids,
        ])
        
        # 截断到最大长度
        if len(input_ids) > self.max_seq_len:
            # 优先保留response部分，截断prompt
            response_len = len(response_ids)
            prompt_max = self.max_seq_len - response_len
            if prompt_max < 50:
                prompt_max = 50
                response_max = self.max_seq_len - 50
                response_ids = response_ids[:response_max]
                labels_response = response_ids
            else:
                labels_response = response_ids
            
            prompt_ids = prompt_ids[-prompt_max:]
            input_ids = torch.cat([prompt_ids, labels_response])
            labels = torch.cat([
                torch.full((len(prompt_ids),), -100, dtype=torch.long),
                labels_response,
            ])
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "weight": torch.tensor(weight, dtype=torch.float32),
        }


def collate_fn(batch):
    """自定义collate函数，处理变长序列"""
    max_len = max(b["input_ids"].shape[0] for b in batch)
    
    input_ids_list = []
    labels_list = []
    attention_mask_list = []
    weights_list = []
    
    for b in batch:
        seq_len = b["input_ids"].shape[0]
        pad_len = max_len - seq_len
        
        # 左填充（与tokenizer配置一致）
        input_ids_list.append(torch.cat([
            torch.zeros(pad_len, dtype=torch.long),
            b["input_ids"]
        ]))
        labels_list.append(torch.cat([
            torch.full((pad_len,), -100, dtype=torch.long),
            b["labels"]
        ]))
        attention_mask_list.append(torch.cat([
            torch.zeros(pad_len, dtype=torch.long),
            torch.ones(seq_len, dtype=torch.long),
        ]))
        weights_list.append(b["weight"])
    
    return {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
        "attention_mask": torch.stack(attention_mask_list),
        "weights": torch.stack(weights_list),
    }


def compute_weighted_loss(logits, labels, weights, ignore_index=-100):
    """
    计算加权因果语言建模损失（论文核心损失函数）
    
    仅对Response部分（labels != -100）计算损失
    按样本循环一致性权重加权
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # 移位: 预测位置n的token用位置n-1的logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # 计算每个token的损失
    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")
    per_token_loss = loss_fct(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
    ).view(batch_size, -1)  # [batch, seq-1]
    
    # 计算每个样本的平均损失（只有非-100的token才有损失）
    mask = (shift_labels != ignore_index).float()
    token_count = mask.sum(dim=1).clamp(min=1)
    per_sample_loss = (per_token_loss * mask).sum(dim=1) / token_count  # [batch]
    
    # 按循环一致性权重加权
    weights = weights.to(per_sample_loss.device)
    # 归一化权重（避免数值不稳定）
    weights = weights / weights.mean()
    
    weighted_loss = (per_sample_loss * weights).mean()
    
    return weighted_loss


def load_base_model_for_training(model_path: str, config: TrainingConfig):
    """加载基座模型用于LoRA训练（使用float16）"""
    logger.info(f"加载基座模型用于训练: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 使用4-bit QLoRA加载以节省显存
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 为k-bit训练准备模型（关键步骤）
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    # 添加LoRA适配器（严格对齐论文参数）
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.target_modules),
        lora_dropout=config.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return tokenizer, model


def load_sft_data(sft_path: str) -> List[Dict]:
    """加载SFT数据集"""
    samples = []
    with open(sft_path, "r") as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples


def load_val_data(val_path: str, n_samples: int = 300) -> List[Dict]:
    """加载验证集数据"""
    samples = []
    with open(val_path, "r") as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    
    # 均衡采样：正确和错误各一半
    correct = [s for s in samples if s["is_correct"]]
    incorrect = [s for s in samples if not s["is_correct"]]
    
    n_each = n_samples // 2
    selected = (
        random.sample(correct, min(n_each, len(correct)))
        + random.sample(incorrect, min(n_each, len(incorrect)))
    )
    random.shuffle(selected)
    return selected


def evaluate_on_val(model, tokenizer, val_samples: List[Dict], device: str = "cuda") -> float:
    """在验证集上评估学生答案预测准确率"""
    from step3_mistake_generate import SFT_SYSTEM_PROMPT, SFT_USER_PROMPT
    
    model.eval()
    correct_count = 0
    total = 0
    
    batch_size = 4
    for i in range(0, min(len(val_samples), 200), batch_size):
        batch = val_samples[i: i + batch_size]
        prompts = []
        ground_truths = []
        
        for s in batch:
            user_content = SFT_USER_PROMPT.format(
                history_text=s["history_text"],
                new_question=s["new_question"],
                new_opt_a=s["new_opt_a"],
                new_opt_b=s["new_opt_b"],
                new_opt_c=s["new_opt_c"],
                new_opt_d=s["new_opt_d"],
            )
            prompt = f"<|im_start|>system\n{SFT_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
            prompts.append(prompt)
            ground_truths.append(s["student_answer"])
        
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        input_len = inputs["input_ids"].shape[1]
        for j, output in enumerate(outputs):
            generated = tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()
            
            # 解析predicted_student_option
            predicted = None
            import re, json as json_lib
            try:
                json_match = re.search(r'\{.*\}', generated, re.DOTALL)
                if json_match:
                    parsed = json_lib.loads(json_match.group())
                    predicted = parsed.get("predicted_student_option", "").upper()
            except:
                pass
            
            if predicted is None:
                # 回退：直接查找选项字母
                match = re.search(r'"predicted_student_option":\s*"([ABCD])"', generated)
                if match:
                    predicted = match.group(1)
            
            if predicted and predicted in "ABCD" and predicted == ground_truths[j]:
                correct_count += 1
            total += 1
    
    accuracy = correct_count / total if total > 0 else 0.0
    model.train()
    return accuracy


def train_one_iteration(
    model,
    tokenizer,
    train_samples: List[Dict],
    val_samples: List[Dict],
    config: TrainingConfig,
    iteration: int,
    save_dir: Path,
) -> Tuple[float, str]:
    """
    执行一次迭代训练（对应Algorithm2中的一次SFT）
    
    Returns: (best_val_accuracy, best_model_path)
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"迭代 {iteration+1}/{config.max_iterations} 开始训练")
    logger.info(f"训练样本数: {len(train_samples)}")
    
    # 创建数据集
    dataset = SFTDataset(train_samples, tokenizer, config.max_seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    total_steps = len(dataloader) * config.max_epochs_per_iter
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    best_val_acc = 0.0
    best_epoch = -1
    no_improve_count = 0
    best_model_path = str(save_dir / f"iter{iteration+1}_best")
    
    device = next(model.parameters()).device
    scaler = torch.cuda.amp.GradScaler(enabled=config.fp16)
    
    for epoch in range(config.max_epochs_per_iter):
        model.train()
        total_loss = 0.0
        step_count = 0
        optimizer.zero_grad()
        
        epoch_start = time.time()
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            weights = batch["weights"]
            
            with torch.cuda.amp.autocast(enabled=config.fp16, dtype=torch.float16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=None,  # 不用内置损失，手动计算加权损失
                )
                logits = outputs.logits
                loss = compute_weighted_loss(logits, labels, weights)
                loss = loss / config.gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (step + 1) % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    config.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                step_count += 1
            
            total_loss += loss.item() * config.gradient_accumulation_steps
            
            if (step + 1) % 50 == 0:
                avg_loss = total_loss / (step + 1)
                elapsed = time.time() - epoch_start
                logger.info(
                    f"  Iter{iteration+1} Epoch{epoch+1} Step{step+1}/{len(dataloader)} "
                    f"Loss={avg_loss:.4f} Time={elapsed:.0f}s"
                )
        
        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - epoch_start
        logger.info(f"Iter{iteration+1} Epoch{epoch+1} 完成: Loss={avg_loss:.4f}, 耗时={epoch_time:.0f}s")
        
        # 验证集评估
        logger.info("验证集评估中...")
        val_acc = evaluate_on_val(model, tokenizer, val_samples)
        logger.info(f"验证集准确率: {val_acc:.4f}")
        
        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve_count = 0
            
            os.makedirs(best_model_path, exist_ok=True)
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            
            with open(Path(best_model_path) / "training_info.json", "w") as f:
                json.dump({
                    "iteration": iteration + 1,
                    "best_epoch": epoch + 1,
                    "val_accuracy": best_val_acc,
                    "train_loss": avg_loss,
                }, f, indent=2)
            
            logger.info(f"保存最优模型 (val_acc={best_val_acc:.4f}) 到: {best_model_path}")
        else:
            no_improve_count += 1
            logger.info(f"验证集无提升，连续{no_improve_count}轮")
        
        # 早停
        if no_improve_count >= config.early_stop_patience:
            logger.info(f"早停触发: 连续{config.early_stop_patience}轮无提升")
            break
    
    logger.info(f"迭代{iteration+1}完成: 最优val_acc={best_val_acc:.4f} (epoch{best_epoch+1})")
    return best_val_acc, best_model_path


def regenerate_sft_data_with_model(
    model,
    tokenizer,
    base_sft_data: List[Dict],
    iteration: int,
) -> List[Dict]:
    """
    使用当前迭代模型重新生成更高质量的SFT数据
    （实现Algorithm2的迭代数据集更新）
    
    核心思路：用当前模型对训练数据重新做循环一致性检验
    优化权重分配，过滤出更高质量的样本
    """
    logger.info(f"使用迭代{iteration}模型更新SFT数据集...")
    
    # 这里使用简化策略：对当前数据集的权重进行动态更新
    # 实际论文中是重新运行完整的MISTAKE-GENERATE流程
    # 由于时间约束，我们采用re-weighting策略
    
    from step3_mistake_generate import (
        CYCLE_FORWARD_PROMPT, CYCLE_BACKWARD_PROMPT,
        batch_generate, parse_option_from_text, extract_misconception,
    )
    
    model.eval()
    
    # 只重新检验错误样本（正确样本保持不变）
    incorrect_samples = [s for s in base_sft_data if not s.get("is_correct", False)]
    correct_samples = [s for s in base_sft_data if s.get("is_correct", False)]
    
    # 采样子集重新检验
    n_recheck = min(2000, len(incorrect_samples))
    recheck_samples = random.sample(incorrect_samples, n_recheck)
    
    # 构建循环一致性校验所需的格式
    # 需要从prompt中提取question信息（简化：直接用原始字段）
    recheck_with_full = []
    for s in recheck_samples:
        recheck_with_full.append({
            "new_question": s.get("new_question", ""),
            "new_opt_a": s.get("new_opt_a", ""),
            "new_opt_b": s.get("new_opt_b", ""),
            "new_opt_c": s.get("new_opt_c", ""),
            "new_opt_d": s.get("new_opt_d", ""),
            "student_answer": s["student_answer"],
            "correct_answer": s["correct_answer"],
            "history_text": s.get("history_text", ""),
            "user_id": s.get("user_id", ""),
            "question_id": s.get("question_id", ""),
            "is_correct": False,
            "misconception": s.get("misconception", ""),
            "original_weight": s.get("weight", 1),
        })
    
    # 执行反向校验（只做backward，用已有misconception）
    backward_prompts = []
    for s in recheck_with_full:
        opts = {"A": s["new_opt_a"], "B": s["new_opt_b"], 
                "C": s["new_opt_c"], "D": s["new_opt_d"]}
        backward_prompts.append(
            CYCLE_BACKWARD_PROMPT.format(
                question=s["new_question"],
                opt_a=opts["A"], opt_b=opts["B"],
                opt_c=opts["C"], opt_d=opts["D"],
                misconception=s["misconception"],
            )
        )
    
    backward_outputs = batch_generate(
        tokenizer, model, backward_prompts,
        max_new_tokens=5, batch_size=16, temperature=0.0
    )
    
    simulated_answers = [parse_option_from_text(o) for o in backward_outputs]
    
    # 更新权重
    updated_incorrect = []
    for s, a_sim in zip(recheck_with_full, simulated_answers):
        weight = s["original_weight"]
        if a_sim == s["student_answer"]:
            weight = max(weight, 2)  # 验证通过，保持或提升权重
        elif a_sim == s["correct_answer"]:
            weight = 0  # 反向检验失败，丢弃
        
        updated_s = dict(s)
        updated_s["weight"] = weight
        updated_s["simulated_answer_iter"] = a_sim
        updated_incorrect.append(updated_s)
    
    # 过滤并合并
    valid_updated = [s for s in updated_incorrect if s["weight"] > 0]
    valid_updated.sort(key=lambda x: x["weight"], reverse=True)
    
    # 重建SFT数据
    new_sft_data = []
    for s in valid_updated[:int(len(base_sft_data) * 0.7)]:
        prompt_s = {
            "history_text": s.get("history_text", ""),
            "new_question": s["new_question"],
            "new_opt_a": s["new_opt_a"],
            "new_opt_b": s["new_opt_b"],
            "new_opt_c": s["new_opt_c"],
            "new_opt_d": s["new_opt_d"],
        }
        from step3_mistake_generate import build_sft_prompt, build_sft_response
        prompt = build_sft_prompt(prompt_s)
        response = build_sft_response(
            s["misconception"], s["student_answer"], s["correct_answer"]
        )
        new_sft_data.append({
            "prompt": prompt,
            "response": response,
            "weight": s["weight"],
            "weight_type": f"iter{iteration}_rechecked",
            "student_answer": s["student_answer"],
            "correct_answer": s["correct_answer"],
            "simulated_answer": s.get("simulated_answer_iter"),
            "misconception": s["misconception"],
            "is_correct": False,
            "new_question": s["new_question"],
            "new_opt_a": s["new_opt_a"],
            "new_opt_b": s["new_opt_b"],
            "new_opt_c": s["new_opt_c"],
            "new_opt_d": s["new_opt_d"],
        })
    
    for s in correct_samples:
        new_sft_data.append(s)
    
    new_sft_data.sort(key=lambda x: x["weight"], reverse=True)
    logger.info(f"更新后SFT数据集: {len(new_sft_data)} 条")
    
    model.train()
    return new_sft_data


def main():
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("步骤4: MISTAKE-UPDATE 迭代式LoRA微调")
    logger.info("=" * 60)
    
    config = TrainingConfig()
    
    # ===== 加载SFT数据集 =====
    sft_path = SFT_DATASET_DIR / "sft_v2.jsonl"
    if not sft_path.exists():
        # 回退到旧版SFT数据
        sft_path = SFT_DATASET_DIR / "sft_data.jsonl"
        logger.warning(f"sft_v2.jsonl不存在，使用备用: {sft_path}")
    
    sft_data = load_sft_data(str(sft_path))
    logger.info(f"加载SFT数据: {len(sft_data)} 条")
    
    # ===== 加载验证集 =====
    val_samples = load_val_data(
        str(PROCESSED_DATA_DIR / "val.jsonl"), config.val_eval_samples
    )
    logger.info(f"验证集样本: {len(val_samples)} 条")
    
    # ===== 加载模型（用于训练）=====
    tokenizer, model = load_base_model_for_training(str(MODEL_PATH), config)
    
    # ===== 迭代训练（Algorithm2）=====
    best_overall_acc = 0.0
    best_overall_model_path = None
    
    current_sft_data = sft_data
    
    for iteration in range(config.max_iterations):
        iter_start = time.time()
        
        # 取权重最高的N条训练样本
        train_sorted = sorted(current_sft_data, key=lambda x: x["weight"], reverse=True)
        train_samples = train_sorted[:config.train_top_n]
        
        logger.info(
            f"\n{'='*60}\n"
            f"迭代 {iteration+1}/{config.max_iterations}\n"
            f"训练样本: {len(train_samples)}\n"
            f"权重分布: {dict((w, sum(1 for s in train_samples if s['weight']==w)) for w in set(s['weight'] for s in train_samples))}"
        )
        
        # 训练一轮
        save_dir = TRAINED_MODEL_DIR
        val_acc, model_path = train_one_iteration(
            model, tokenizer, train_samples, val_samples, config, iteration, save_dir
        )
        
        # 更新全局最优
        if val_acc > best_overall_acc:
            best_overall_acc = val_acc
            best_overall_model_path = model_path
            
            # 复制到final best model路径
            final_best_path = TRAINED_MODEL_DIR / "best_model_v2"
            if final_best_path.exists():
                shutil.rmtree(final_best_path)
            shutil.copytree(model_path, final_best_path)
            logger.info(f"更新全局最优模型: val_acc={best_overall_acc:.4f}")
        
        iter_time = time.time() - iter_start
        logger.info(f"迭代{iteration+1}总耗时: {iter_time/60:.1f}分钟")
        
        # 检查时间限制（预留评估时间）
        elapsed = time.time() - start_time
        if elapsed > 7200:  # 2小时限制
            logger.warning("接近时间限制，提前结束训练迭代")
            break
        
        # 如果不是最后一轮，更新数据集
        if iteration < config.max_iterations - 1:
            logger.info("使用当前模型更新SFT数据集...")
            try:
                current_sft_data = regenerate_sft_data_with_model(
                    model, tokenizer, current_sft_data, iteration + 1
                )
                # 保存更新后的数据集
                updated_path = SFT_DATASET_DIR / f"sft_v2_iter{iteration+1}.jsonl"
                with open(updated_path, "w") as f:
                    for item in current_sft_data:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                logger.info(f"更新数据集已保存: {updated_path}")
            except Exception as e:
                logger.error(f"数据集更新失败 (iteration {iteration+1}): {e}")
                logger.info("继续使用原始SFT数据集")
    
    # ===== 保存最终模型 =====
    final_model_path = TRAINED_MODEL_DIR / "final_model"
    if best_overall_model_path and Path(best_overall_model_path).exists():
        if not final_model_path.exists():
            shutil.copytree(best_overall_model_path, final_model_path)
    
    # ===== 保存训练配置 =====
    config_path = TRAINED_MODEL_DIR / "training_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "max_iterations": config.max_iterations,
            "max_epochs_per_iter": config.max_epochs_per_iter,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "effective_batch_size": config.batch_size * config.gradient_accumulation_steps,
            "max_seq_len": config.max_seq_len,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "target_modules": list(config.target_modules),
            "train_top_n": config.train_top_n,
            "best_val_accuracy": best_overall_acc,
            "best_model_path": str(best_overall_model_path or ""),
            "base_model": str(MODEL_PATH),
        }, f, indent=2)
    
    elapsed = time.time() - start_time
    logger.info(f"\n步骤4完成! 总耗时: {elapsed/60:.1f}分钟")
    logger.info(f"最优验证集准确率: {best_overall_acc:.4f}")
    logger.info(f"最优模型路径: {best_overall_model_path}")
    
    return str(best_overall_model_path or "")


if __name__ == "__main__":
    main()
