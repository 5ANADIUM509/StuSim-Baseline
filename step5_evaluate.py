#!/usr/bin/env python3
"""
步骤5: 全维度效果评估
严格执行SFT前/后模型的单变量对比实验

核心指标:
1. 整体预测准确率
2. 学生正确作答子集准确率
3. 学生错误作答子集准确率
4. 正确答案预测准确率
5. 循环一致性通过率
6. 各子集详细统计
"""

import os, sys, json, random, logging, time, re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, Counter
from datetime import datetime

# ========== 强制数据盘路径 ==========
os.environ["HF_HOME"] = "/root/autodl-tmp/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "/root/autodl-tmp/huggingface_cache"
os.environ["TORCH_HOME"] = "/root/autodl-tmp/torch_cache"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ========== 路径配置 ==========
BASE_DIR = Path("/root/autodl-tmp")
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
TRAINED_MODEL_DIR = BASE_DIR / "trained_model"
EVAL_REPORT_DIR = BASE_DIR / "evaluation_report"
LOG_DIR = BASE_DIR / "logs"
MODEL_PATH = BASE_DIR / "models/Qwen/Qwen2.5-3B-Instruct"
TEMP_DIR = BASE_DIR / "temp"

EVAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ========== 日志配置 ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "step5_evaluate.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ========== 随机种子 ==========
random.seed(42)

# ========== 评估配置 ==========
EVAL_SAMPLE_SIZE = 500   # 总评估样本数
EVAL_BATCH_SIZE = 4      # 推理批大小

# ========== Prompt模板（与训练时完全一致）==========
SFT_SYSTEM_PROMPT = """You are an expert mathematics educator specializing in student misconception analysis.

CRITICAL CONSTRAINT: Any inferred student misconception must be restorable - 
using that misconception to simulate solving must produce the student's actual chosen answer."""

SFT_USER_PROMPT = """Based on this student's 30-question history, identify their misconception pattern and predict their answer.

Student History:
{history_text}

New Question: {new_question}
A. {new_opt_a}
B. {new_opt_b}
C. {new_opt_c}
D. {new_opt_d}

Analyze the student's recurring error patterns from their history, then predict:
1. The main misconception driving their errors
2. Which option they will MOST LIKELY choose (based on their history patterns, not what's correct)
3. Which option is actually correct

Output ONLY valid JSON:
{{"misconception_pattern": "one sentence describing the main pattern", "predicted_student_option": "X", "predicted_correct_option": "Y"}}"""

# 反向验证prompt（循环一致性通过率计算用）
CYCLE_BACKWARD_PROMPT = """Question: {question}
Options: A) {opt_a}  B) {opt_b}  C) {opt_c}  D) {opt_d}

A student has this misconception: {misconception}

Based ONLY on this misconception, which option would this student choose?
Reply with ONLY a single letter: A, B, C, or D

Answer:"""


def load_model_4bit(model_path: str, lora_path: Optional[str] = None):
    """以4-bit NF4量化加载模型（可选加载LoRA适配器）"""
    logger.info(f"加载模型: {model_path}")
    if lora_path:
        logger.info(f"加载LoRA适配器: {lora_path}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    if lora_path and Path(lora_path).exists():
        model = PeftModel.from_pretrained(model, lora_path)
        logger.info("LoRA适配器加载成功")
    
    model.eval()
    return tokenizer, model


def load_test_data() -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """加载测试数据集（完整集、正确子集、错误子集）"""
    test_path = PROCESSED_DATA_DIR / "test.jsonl"
    
    all_test = []
    with open(test_path, "r") as f:
        for line in f:
            all_test.append(json.loads(line.strip()))
    
    correct_subset = [s for s in all_test if s["is_correct"]]
    incorrect_subset = [s for s in all_test if not s["is_correct"]]
    
    logger.info(f"测试集总样本: {len(all_test)}")
    logger.info(f"正确子集: {len(correct_subset)}, 错误子集: {len(incorrect_subset)}")
    
    return all_test, correct_subset, incorrect_subset


def sample_evaluation_set(
    correct_subset: List[Dict],
    incorrect_subset: List[Dict],
    total_size: int = EVAL_SAMPLE_SIZE,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    均衡采样评估集（保持正确/错误比例约50/50）
    同时保证两组实验的样本完全一致
    """
    n_each = total_size // 2
    
    selected_correct = random.sample(correct_subset, min(n_each, len(correct_subset)))
    selected_incorrect = random.sample(incorrect_subset, min(n_each, len(incorrect_subset)))
    
    all_selected = selected_correct + selected_incorrect
    random.shuffle(all_selected)
    
    logger.info(f"评估集: 总计{len(all_selected)} (正确{len(selected_correct)} + 错误{len(selected_incorrect)})")
    return all_selected, selected_correct, selected_incorrect


def build_inference_prompt(sample: Dict) -> str:
    """构建推理Prompt（与训练时完全一致）"""
    user_content = SFT_USER_PROMPT.format(
        history_text=sample["history_text"],
        new_question=sample["new_question"],
        new_opt_a=sample["new_opt_a"],
        new_opt_b=sample["new_opt_b"],
        new_opt_c=sample["new_opt_c"],
        new_opt_d=sample["new_opt_d"],
    )
    return f"<|im_start|>system\n{SFT_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"


def parse_model_output(generated_text: str) -> Dict[str, Optional[str]]:
    """解析模型输出，提取predicted_student_option和predicted_correct_option"""
    result = {
        "predicted_student_option": None,
        "predicted_correct_option": None,
        "misconception_pattern": None,
        "raw_output": generated_text,
        "parse_success": False,
    }
    
    # 方法1: 解析完整JSON
    try:
        json_match = re.search(r'\{[^{}]*\}', generated_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            student_opt = parsed.get("predicted_student_option", "")
            correct_opt = parsed.get("predicted_correct_option", "")
            
            if student_opt.upper() in "ABCD" and len(student_opt) == 1:
                result["predicted_student_option"] = student_opt.upper()
            if correct_opt.upper() in "ABCD" and len(correct_opt) == 1:
                result["predicted_correct_option"] = correct_opt.upper()
            result["misconception_pattern"] = parsed.get("misconception_pattern", "")
            
            if result["predicted_student_option"] and result["predicted_correct_option"]:
                result["parse_success"] = True
                return result
    except Exception:
        pass
    
    # 方法2: 正则提取
    student_match = re.search(
        r'"predicted_student_option"\s*:\s*"([ABCD])"', generated_text, re.IGNORECASE
    )
    correct_match = re.search(
        r'"predicted_correct_option"\s*:\s*"([ABCD])"', generated_text, re.IGNORECASE
    )
    
    if student_match:
        result["predicted_student_option"] = student_match.group(1).upper()
    if correct_match:
        result["predicted_correct_option"] = correct_match.group(1).upper()
    
    if result["predicted_student_option"] and result["predicted_correct_option"]:
        result["parse_success"] = True
    
    return result


def run_batch_inference(
    tokenizer,
    model,
    samples: List[Dict],
    batch_size: int = EVAL_BATCH_SIZE,
    cache_path: Optional[str] = None,
) -> List[Dict]:
    """
    批量推理，返回包含预测结果的样本列表
    支持推理缓存以避免重复计算
    """
    # 加载缓存
    if cache_path and Path(cache_path).exists():
        logger.info(f"加载推理缓存: {cache_path}")
        with open(cache_path, "r") as f:
            cached = [json.loads(l) for l in f]
        if len(cached) == len(samples):
            logger.info(f"缓存命中，跳过推理")
            return cached
    
    results = []
    total = len(samples)
    parse_success_count = 0
    
    logger.info(f"开始批量推理，共 {total} 个样本...")
    
    for i in range(0, total, batch_size):
        batch = samples[i: i + batch_size]
        prompts = [build_inference_prompt(s) for s in batch]
        
        # Tokenize
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(next(model.parameters()).device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        input_len = inputs["input_ids"].shape[1]
        
        for j, (sample, output) in enumerate(zip(batch, outputs)):
            generated = tokenizer.decode(
                output[input_len:], skip_special_tokens=True
            ).strip()
            
            parsed = parse_model_output(generated)
            
            result = {
                "user_id": sample["user_id"],
                "question_id": sample["question_id"],
                "student_answer": sample["student_answer"],
                "correct_answer": sample["correct_answer"],
                "is_correct": sample["is_correct"],
                "predicted_student_option": parsed["predicted_student_option"],
                "predicted_correct_option": parsed["predicted_correct_option"],
                "misconception_pattern": parsed.get("misconception_pattern", ""),
                "raw_output": generated[:300],
                "parse_success": parsed["parse_success"],
                "new_question": sample["new_question"],
                "new_opt_a": sample["new_opt_a"],
                "new_opt_b": sample["new_opt_b"],
                "new_opt_c": sample["new_opt_c"],
                "new_opt_d": sample["new_opt_d"],
            }
            results.append(result)
            
            if parsed["parse_success"]:
                parse_success_count += 1
        
        if (i // batch_size + 1) % 10 == 0:
            logger.info(
                f"  推理进度: {min(i+batch_size, total)}/{total}, "
                f"解析成功率: {parse_success_count/len(results):.2%}"
            )
    
    # 保存缓存
    if cache_path:
        with open(cache_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    logger.info(f"推理完成，解析成功率: {parse_success_count/total:.2%}")
    return results


def check_cycle_consistency(
    results: List[Dict],
    tokenizer,
    model,
) -> float:
    """
    计算循环一致性通过率
    
    对每个有效预测结果，检验：
    用推断的错误概念反向模拟是否能得到学生真实答案
    """
    logger.info("计算循环一致性通过率...")
    
    valid_results = [
        r for r in results 
        if r["parse_success"] and r.get("misconception_pattern")
        and not r["is_correct"]  # 只对错误样本做循环一致性检验
    ]
    
    if not valid_results:
        return 0.0
    
    # 构建反向验证prompts
    backward_prompts = []
    for r in valid_results[:200]:  # 限制数量
        opts = {
            "A": r["new_opt_a"], "B": r["new_opt_b"],
            "C": r["new_opt_c"], "D": r["new_opt_d"],
        }
        backward_prompts.append(
            CYCLE_BACKWARD_PROMPT.format(
                question=r["new_question"],
                opt_a=opts["A"], opt_b=opts["B"],
                opt_c=opts["C"], opt_d=opts["D"],
                misconception=r.get("misconception_pattern", ""),
            )
        )
    
    # 批量推理
    all_backward_outputs = []
    batch_size = 16
    for i in range(0, len(backward_prompts), batch_size):
        batch = backward_prompts[i: i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(next(model.parameters()).device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        input_len = inputs["input_ids"].shape[1]
        for output in outputs:
            generated = tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()
            # 提取字母
            match = re.search(r'[ABCD]', generated.upper())
            all_backward_outputs.append(match.group() if match else None)
    
    # 计算通过率
    pass_count = 0
    total_checked = min(len(valid_results), 200)
    for i, (r, sim_ans) in enumerate(zip(valid_results[:200], all_backward_outputs)):
        if sim_ans == r["student_answer"]:
            pass_count += 1
    
    cycle_consistency_rate = pass_count / total_checked if total_checked > 0 else 0.0
    logger.info(f"循环一致性通过率: {cycle_consistency_rate:.4f} ({pass_count}/{total_checked})")
    
    return cycle_consistency_rate


def compute_all_metrics(
    results: List[Dict],
    label: str = "",
) -> Dict:
    """
    计算所有评估指标
    
    只统计parse_success=True的样本，确保两组实验样本完全一致
    """
    # 过滤解析成功的样本
    valid = [r for r in results if r["parse_success"]]
    
    if not valid:
        logger.warning(f"[{label}] 无有效预测结果！")
        return {}
    
    # 分割子集
    correct_subset = [r for r in valid if r["is_correct"]]
    incorrect_subset = [r for r in valid if not r["is_correct"]]
    
    def calc_acc(subset, metric="student"):
        if not subset:
            return 0.0, 0, 0
        if metric == "student":
            hits = sum(1 for r in subset if r["predicted_student_option"] == r["student_answer"])
        else:  # correct_answer
            hits = sum(1 for r in subset if r["predicted_correct_option"] == r["correct_answer"])
        return hits / len(subset), hits, len(subset)
    
    # 核心指标
    overall_acc, overall_hits, overall_total = calc_acc(valid, "student")
    correct_sub_acc, correct_hits, correct_total = calc_acc(correct_subset, "student")
    incorrect_sub_acc, incorrect_hits, incorrect_total = calc_acc(incorrect_subset, "student")
    
    # 辅助指标
    correct_ans_acc, _, _ = calc_acc(valid, "correct")
    
    # 解析率
    parse_rate = len(valid) / len(results) if results else 0.0
    
    metrics = {
        "label": label,
        "total_samples": len(results),
        "valid_samples": len(valid),
        "parse_success_rate": parse_rate,
        # 核心指标
        "overall_accuracy": overall_acc,
        "overall_hits": overall_hits,
        "overall_total": overall_total,
        "correct_subset_accuracy": correct_sub_acc,
        "correct_subset_hits": correct_hits,
        "correct_subset_total": correct_total,
        "incorrect_subset_accuracy": incorrect_sub_acc,
        "incorrect_subset_hits": incorrect_hits,
        "incorrect_subset_total": incorrect_total,
        # 辅助指标
        "correct_answer_prediction_accuracy": correct_ans_acc,
    }
    
    logger.info(f"\n[{label}] 评估结果:")
    logger.info(f"  整体预测准确率: {overall_acc:.4f} ({overall_hits}/{overall_total})")
    logger.info(f"  学生正确子集准确率: {correct_sub_acc:.4f} ({correct_hits}/{correct_total})")
    logger.info(f"  学生错误子集准确率: {incorrect_sub_acc:.4f} ({incorrect_hits}/{incorrect_total})")
    logger.info(f"  正确答案预测准确率: {correct_ans_acc:.4f}")
    logger.info(f"  解析成功率: {parse_rate:.4f}")
    
    return metrics


def align_results(base_results: List[Dict], ft_results: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    对齐两组实验的有效样本，确保比较公平
    只保留两组都解析成功的样本
    """
    base_success = {(r["user_id"], r["question_id"]): r for r in base_results if r["parse_success"]}
    ft_success = {(r["user_id"], r["question_id"]): r for r in ft_results if r["parse_success"]}
    
    common_keys = set(base_success.keys()) & set(ft_success.keys())
    
    aligned_base = [base_success[k] for k in common_keys]
    aligned_ft = [ft_success[k] for k in common_keys]
    
    logger.info(f"对齐后共同有效样本: {len(common_keys)}")
    return aligned_base, aligned_ft


def generate_markdown_report(
    test_stats: Dict,
    base_metrics: Dict,
    ft_metrics: Dict,
    base_cycle_rate: float,
    ft_cycle_rate: float,
    output_path: str,
):
    """生成标准化Markdown评估报告"""
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def fmt_pct(v):
        return f"{v*100:.2f}%" if v is not None else "N/A"
    
    def fmt_delta(v1, v2):
        if v1 is None or v2 is None:
            return "N/A"
        delta = (v2 - v1) * 100
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        return f"{arrow} {abs(delta):.2f}%"
    
    report = f"""# MISTAKE方法效果评估报告

> 生成时间: {now}  
> 方法: MISTAKE-CYCLE+CORRECT 强约束变体  
> 基座模型: Qwen2.5-3B-Instruct  
> 数据集: EEDI数学教育数据集  

---

## 一、测试集基础信息

| 指标 | 数值 |
|------|------|
| 测试集总样本数 | {test_stats.get('test_total', 'N/A')} |
| 学生正确作答子集 | {test_stats.get('test_correct', 'N/A')} |
| 学生错误作答子集 | {test_stats.get('test_incorrect', 'N/A')} |
| 本次评估样本数 | {base_metrics.get('total_samples', 'N/A')} |
| 两组对齐有效样本数 | {base_metrics.get('overall_total', 'N/A')} |

---

## 二、核心预测准确率对比（三类核心指标）

### 2.1 整体预测准确率

| 模型 | 准确率 | 命中/总数 |
|------|--------|-----------|
| SFT前（基座模型） | {fmt_pct(base_metrics.get('overall_accuracy'))} | {base_metrics.get('overall_hits', 'N/A')}/{base_metrics.get('overall_total', 'N/A')} |
| SFT后（微调模型） | {fmt_pct(ft_metrics.get('overall_accuracy'))} | {ft_metrics.get('overall_hits', 'N/A')}/{ft_metrics.get('overall_total', 'N/A')} |
| **Cycle-Consistency提升** | **{fmt_delta(base_metrics.get('overall_accuracy'), ft_metrics.get('overall_accuracy'))}** | — |

### 2.2 学生正确作答子集准确率

| 模型 | 准确率 | 命中/总数 |
|------|--------|-----------|
| SFT前（基座模型） | {fmt_pct(base_metrics.get('correct_subset_accuracy'))} | {base_metrics.get('correct_subset_hits', 'N/A')}/{base_metrics.get('correct_subset_total', 'N/A')} |
| SFT后（微调模型） | {fmt_pct(ft_metrics.get('correct_subset_accuracy'))} | {ft_metrics.get('correct_subset_hits', 'N/A')}/{ft_metrics.get('correct_subset_total', 'N/A')} |
| **提升** | **{fmt_delta(base_metrics.get('correct_subset_accuracy'), ft_metrics.get('correct_subset_accuracy'))}** | — |

### 2.3 学生错误作答子集准确率

| 模型 | 准确率 | 命中/总数 |
|------|--------|-----------|
| SFT前（基座模型） | {fmt_pct(base_metrics.get('incorrect_subset_accuracy'))} | {base_metrics.get('incorrect_subset_hits', 'N/A')}/{base_metrics.get('incorrect_subset_total', 'N/A')} |
| SFT后（微调模型） | {fmt_pct(ft_metrics.get('incorrect_subset_accuracy'))} | {ft_metrics.get('incorrect_subset_hits', 'N/A')}/{ft_metrics.get('incorrect_subset_total', 'N/A')} |
| **提升** | **{fmt_delta(base_metrics.get('incorrect_subset_accuracy'), ft_metrics.get('incorrect_subset_accuracy'))}** | — |

---

## 三、辅助指标对比

| 指标 | SFT前 | SFT后 | 变化 |
|------|-------|-------|------|
| 正确答案预测准确率 | {fmt_pct(base_metrics.get('correct_answer_prediction_accuracy'))} | {fmt_pct(ft_metrics.get('correct_answer_prediction_accuracy'))} | {fmt_delta(base_metrics.get('correct_answer_prediction_accuracy'), ft_metrics.get('correct_answer_prediction_accuracy'))} |
| 循环一致性通过率 | {fmt_pct(base_cycle_rate)} | {fmt_pct(ft_cycle_rate)} | {fmt_delta(base_cycle_rate, ft_cycle_rate)} |
| 输出解析成功率 | {fmt_pct(base_metrics.get('parse_success_rate'))} | {fmt_pct(ft_metrics.get('parse_success_rate'))} | {fmt_delta(base_metrics.get('parse_success_rate'), ft_metrics.get('parse_success_rate'))} |

---

## 四、效果分析

### 4.1 Cycle-Consistency方法核心价值

本实验实现了论文MISTAKE-CYCLE+CORRECT强约束变体:
- **正向推断**: 输入(题目, 学生错误答案) → 推断错误概念
- **反向模拟**: 输入(题目, 错误概念) → 模拟学生答案  
- **权重赋值**: 模拟答案=真实答案(α=2)、模拟≠正确≠真实(α=1)、模拟=正确答案(丢弃)

### 4.2 与论文结论对齐验证

| 论文预期 | 本实验结果 | 是否对齐 |
|----------|------------|----------|
| SFT提升整体预测准确率 | {fmt_pct(base_metrics.get('overall_accuracy'))} → {fmt_pct(ft_metrics.get('overall_accuracy'))} | {'✅' if (ft_metrics.get('overall_accuracy',0) > base_metrics.get('overall_accuracy',0)) else '⚠️需分析'} |
| 错误子集提升显著 | {fmt_pct(base_metrics.get('incorrect_subset_accuracy'))} → {fmt_pct(ft_metrics.get('incorrect_subset_accuracy'))} | {'✅' if (ft_metrics.get('incorrect_subset_accuracy',0) > base_metrics.get('incorrect_subset_accuracy',0)) else '⚠️需分析'} |
| 循环一致性通过率提升 | {fmt_pct(base_cycle_rate)} → {fmt_pct(ft_cycle_rate)} | {'✅' if ft_cycle_rate > base_cycle_rate else '⚠️需分析'} |

### 4.3 关键发现

- **错误子集提升** 是最核心的验证指标: 模型是否学会预测学生的具体错误选择
- **正确子集准确率** 反映模型对已掌握概念的识别能力
- **循环一致性通过率** 验证错误概念推断的质量和可还原性

---

## 五、实验配置

| 配置项 | 值 |
|--------|-----|
| 基座模型 | Qwen2.5-3B-Instruct |
| 量化方式 | 4-bit NF4 |
| LoRA rank | r=8, lora_alpha=16 |
| LoRA目标模块 | q_proj, k_proj, v_proj, o_proj |
| 迭代轮次 | T=3 |
| 训练数据 | EEDI MISTAKE-CYCLE过滤后 8000条 |
| 推理参数 | greedy decoding, max_new_tokens=100 |
| 随机种子 | 42 |

---

*报告由 step5_evaluate.py 自动生成*
"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    logger.info(f"评估报告已保存到: {output_path}")


def find_best_lora_model() -> Optional[str]:
    """自动查找最优LoRA模型路径"""
    candidates = [
        TRAINED_MODEL_DIR / "best_model_v2",
        TRAINED_MODEL_DIR / "final_model",
        TRAINED_MODEL_DIR / "iter3_best",
        TRAINED_MODEL_DIR / "iter2_best",
        TRAINED_MODEL_DIR / "iter1_best",
        TRAINED_MODEL_DIR / "best_model",
    ]
    
    for path in candidates:
        if path.exists() and (path / "adapter_config.json").exists():
            logger.info(f"找到LoRA模型: {path}")
            return str(path)
    
    logger.warning("未找到LoRA模型，将只评估基座模型")
    return None


def main():
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("步骤5: 全维度效果评估")
    logger.info("=" * 60)
    
    # ===== 加载测试数据 =====
    all_test, correct_subset, incorrect_subset = load_test_data()
    
    test_stats = {
        "test_total": len(all_test),
        "test_correct": len(correct_subset),
        "test_incorrect": len(incorrect_subset),
    }
    
    # 保存测试集统计（如果不存在）
    stats_path = PROCESSED_DATA_DIR / "test_stats.json"
    if not stats_path.exists():
        with open(stats_path, "w") as f:
            json.dump(test_stats, f, indent=2)
    
    # 均衡采样评估集（两组实验用相同的样本）
    eval_samples, eval_correct, eval_incorrect = sample_evaluation_set(
        correct_subset, incorrect_subset, EVAL_SAMPLE_SIZE
    )
    
    # 保存评估集
    eval_set_path = TEMP_DIR / "eval_samples.jsonl"
    with open(eval_set_path, "w") as f:
        for s in eval_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info(f"评估集已保存: {eval_set_path}")
    
    # ===== 阶段1: 评估SFT前基座模型 =====
    logger.info("\n" + "="*40)
    logger.info("阶段1: 评估SFT前基座模型")
    logger.info("="*40)
    
    base_cache_path = str(TEMP_DIR / "base_model_results.jsonl")
    
    tokenizer_base, model_base = load_model_4bit(str(MODEL_PATH), lora_path=None)
    base_results = run_batch_inference(
        tokenizer_base, model_base, eval_samples,
        batch_size=EVAL_BATCH_SIZE, cache_path=base_cache_path
    )
    
    # 计算基座模型循环一致性通过率
    base_cycle_rate = check_cycle_consistency(base_results, tokenizer_base, model_base)
    
    # 释放基座模型内存
    del model_base
    torch.cuda.empty_cache()
    logger.info("基座模型内存已释放")
    
    # ===== 阶段2: 评估SFT后微调模型 =====
    logger.info("\n" + "="*40)
    logger.info("阶段2: 评估SFT后微调模型")
    logger.info("="*40)
    
    best_lora_path = find_best_lora_model()
    ft_cache_path = str(TEMP_DIR / "ft_model_results.jsonl")
    
    tokenizer_ft, model_ft = load_model_4bit(str(MODEL_PATH), lora_path=best_lora_path)
    ft_results = run_batch_inference(
        tokenizer_ft, model_ft, eval_samples,
        batch_size=EVAL_BATCH_SIZE, cache_path=ft_cache_path
    )
    
    # 计算微调模型循环一致性通过率
    ft_cycle_rate = check_cycle_consistency(ft_results, tokenizer_ft, model_ft)
    
    del model_ft
    torch.cuda.empty_cache()
    
    # ===== 阶段3: 对齐两组实验样本 =====
    logger.info("\n" + "="*40)
    logger.info("阶段3: 对齐实验样本，计算指标")
    logger.info("="*40)
    
    aligned_base, aligned_ft = align_results(base_results, ft_results)
    
    # 计算所有指标
    base_metrics = compute_all_metrics(aligned_base, label="SFT前(基座模型)")
    ft_metrics = compute_all_metrics(aligned_ft, label="SFT后(微调模型)")
    
    # ===== 阶段4: 生成评估报告 =====
    report_path = str(EVAL_REPORT_DIR / "evaluation_report_v2.md")
    generate_markdown_report(
        test_stats, base_metrics, ft_metrics,
        base_cycle_rate, ft_cycle_rate,
        report_path
    )
    
    # 保存原始指标数据
    metrics_path = str(EVAL_REPORT_DIR / "metrics_v2.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "base_model": base_metrics,
            "fine_tuned_model": ft_metrics,
            "base_cycle_consistency_rate": base_cycle_rate,
            "ft_cycle_consistency_rate": ft_cycle_rate,
            "test_stats": test_stats,
            "eval_timestamp": datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)
    
    elapsed = time.time() - start_time
    logger.info(f"\n步骤5完成! 总耗时: {elapsed/60:.1f}分钟")
    logger.info(f"评估报告路径: {report_path}")
    
    # 打印核心结果摘要
    logger.info("\n" + "="*60)
    logger.info("核心评估结果摘要")
    logger.info("="*60)
    logger.info(f"整体准确率: {base_metrics.get('overall_accuracy', 0):.2%} → {ft_metrics.get('overall_accuracy', 0):.2%}")
    logger.info(f"正确子集: {base_metrics.get('correct_subset_accuracy', 0):.2%} → {ft_metrics.get('correct_subset_accuracy', 0):.2%}")
    logger.info(f"错误子集: {base_metrics.get('incorrect_subset_accuracy', 0):.2%} → {ft_metrics.get('incorrect_subset_accuracy', 0):.2%}")
    logger.info(f"循环一致性: {base_cycle_rate:.2%} → {ft_cycle_rate:.2%}")


if __name__ == "__main__":
    main()
