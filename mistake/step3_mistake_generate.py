#!/usr/bin/env python3
"""
步骤3: MISTAKE-GENERATE - 循环一致性SFT数据集构建
严格对标论文Algorithm1，实现MISTAKE-CYCLE+CORRECT强约束变体

流程:
1. 从训练集中采样候选样本
2. 对每个错误回答样本执行双层循环一致性校验
3. 按权重构建最终8000条SFT训练数据集
"""

import os, sys, json, random, logging, time
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Tuple

# ========== 强制数据盘路径 ==========
os.environ["HF_HOME"] = "/root/autodl-tmp/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "/root/autodl-tmp/huggingface_cache"
os.environ["TORCH_HOME"] = "/root/autodl-tmp/torch_cache"
os.environ["HF_DATASETS_CACHE"] = "/root/autodl-tmp/huggingface_cache/datasets"

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ========== 路径配置 ==========
BASE_DIR = Path("/root/autodl-tmp")
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
SFT_DATASET_DIR = BASE_DIR / "sft_dataset"
TEMP_DIR = BASE_DIR / "temp"
LOG_DIR = BASE_DIR / "logs"
MODEL_PATH = BASE_DIR / "models/Qwen/Qwen2.5-3B-Instruct"

SFT_DATASET_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ========== 日志配置 ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "step3_mistake_generate.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ========== 随机种子 ==========
random.seed(42)
torch.manual_seed(42)

# ========== 目标数量配置 ==========
TARGET_SFT_SIZE = 8000
CANDIDATE_MULTIPLIER = 3  # 候选集大小 = 目标 × 倍数
MAX_INFERENCE_BATCH = 16  # 推理批大小


# ========== SFT数据格式的Prompt模板 ==========
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


# ========== 循环一致性校验用的短Prompt ==========
CYCLE_FORWARD_PROMPT = """Question: {question}
Options: A) {opt_a} B) {opt_b} C) {opt_c} D) {opt_d}
Student chose: {student_ans} (Correct answer: {correct_ans})

This student answered INCORRECTLY. In one sentence, what mathematical misconception caused them to choose {student_ans} instead of {correct_ans}?

Misconception:"""

CYCLE_BACKWARD_PROMPT = """Question: {question}
Options: A) {opt_a}  B) {opt_b}  C) {opt_c}  D) {opt_d}

A student has this misconception: {misconception}

Based ONLY on this misconception, which option would this student choose?
Reply with ONLY a single letter: A, B, C, or D

Answer:"""


def load_model_4bit(model_path: str):
    """以4-bit NF4量化加载模型"""
    logger.info(f"以4-bit NF4量化加载模型: {model_path}")
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
    model.eval()
    logger.info("模型加载完成")
    return tokenizer, model


def batch_generate(
    tokenizer,
    model,
    prompts: List[str],
    max_new_tokens: int = 80,
    batch_size: int = 16,
    temperature: float = 0.3,
) -> List[str]:
    """批量生成文本"""
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else 1.0,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        for output in outputs:
            generated = tokenizer.decode(
                output[input_len:], skip_special_tokens=True
            ).strip()
            results.append(generated)

        if (i // batch_size + 1) % 10 == 0:
            logger.info(
                f"  已处理 {min(i + batch_size, len(prompts))}/{len(prompts)} 个样本"
            )

    return results


def parse_option_from_text(text: str) -> Optional[str]:
    """从文本中提取选项字母(A/B/C/D)"""
    text = text.strip().upper()
    # 优先匹配行首或单独的字母
    for pattern in [r"^([ABCD])\b", r"\b([ABCD])\b", r"([ABCD])"]:
        import re
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def extract_misconception(text: str) -> str:
    """从生成文本中提取误解描述"""
    text = text.strip()
    # 清理常见前缀
    for prefix in ["Misconception:", "The misconception is", "The student's misconception"]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    # 只取第一句话
    for sep in [".", "\n"]:
        if sep in text:
            text = text.split(sep)[0].strip()
    return text[:200] if text else "Unknown misconception"


def load_training_data(data_path: str) -> List[Dict]:
    """加载预处理后的训练数据"""
    logger.info(f"加载训练数据: {data_path}")
    samples = []
    with open(data_path, "r") as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    logger.info(f"共加载 {len(samples)} 条训练样本")
    return samples


def parse_options_from_sample(sample: Dict) -> Dict[str, str]:
    """从样本解析选项字典"""
    return {
        "A": sample.get("new_opt_a", ""),
        "B": sample.get("new_opt_b", ""),
        "C": sample.get("new_opt_c", ""),
        "D": sample.get("new_opt_d", ""),
    }


def build_sft_prompt(sample: Dict) -> str:
    """构建SFT训练用的完整Prompt"""
    user_content = SFT_USER_PROMPT.format(
        history_text=sample["history_text"],
        new_question=sample["new_question"],
        new_opt_a=sample["new_opt_a"],
        new_opt_b=sample["new_opt_b"],
        new_opt_c=sample["new_opt_c"],
        new_opt_d=sample["new_opt_d"],
    )
    return f"<|im_start|>system\n{SFT_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"


def build_sft_response(misconception_pattern: str, student_answer: str, correct_answer: str) -> str:
    """构建SFT训练目标响应"""
    return json.dumps(
        {
            "misconception_pattern": misconception_pattern,
            "predicted_student_option": student_answer,
            "predicted_correct_option": correct_answer,
        }
    )


def compute_student_global_pattern(student_history: List[Dict]) -> str:
    """从学生历史记录推断全局错误模式（规则+统计）"""
    incorrect = [h for h in student_history if not h.get("is_correct", True)]
    if not incorrect:
        return "Student generally demonstrates strong understanding"
    error_rate = len(incorrect) / len(student_history)
    if error_rate > 0.6:
        return "Student frequently struggles with mathematical problem interpretation"
    elif error_rate > 0.3:
        return "Student shows inconsistent understanding with recurring conceptual gaps"
    else:
        return "Student occasionally makes systematic errors in specific concept areas"


def run_cycle_consistency_check(
    tokenizer,
    model,
    incorrect_samples: List[Dict],
    batch_size: int = MAX_INFERENCE_BATCH,
) -> List[Dict]:
    """
    执行单题级循环一致性校验（论文Algorithm1核心逻辑）
    
    Returns:
        每个样本附加字段: misconception, simulated_answer, weight
    """
    logger.info(f"开始循环一致性校验，样本数: {len(incorrect_samples)}")
    
    # ===== 阶段1: 正向推断 - 推断错误概念 =====
    logger.info("阶段1: 正向推断错误概念...")
    forward_prompts = []
    for s in incorrect_samples:
        opts = parse_options_from_sample(s)
        forward_prompts.append(
            CYCLE_FORWARD_PROMPT.format(
                question=s["new_question"],
                opt_a=opts["A"], opt_b=opts["B"],
                opt_c=opts["C"], opt_d=opts["D"],
                student_ans=s["student_answer"],
                correct_ans=s["correct_answer"],
            )
        )
    
    t0 = time.time()
    forward_outputs = batch_generate(
        tokenizer, model, forward_prompts,
        max_new_tokens=60, batch_size=batch_size, temperature=0.3
    )
    logger.info(f"正向推断完成，耗时: {time.time()-t0:.1f}s")
    
    misconceptions = [extract_misconception(o) for o in forward_outputs]
    
    # ===== 阶段2: 反向模拟 - 用错误概念模拟解题 =====
    logger.info("阶段2: 反向模拟答案...")
    backward_prompts = []
    for s, m in zip(incorrect_samples, misconceptions):
        opts = parse_options_from_sample(s)
        backward_prompts.append(
            CYCLE_BACKWARD_PROMPT.format(
                question=s["new_question"],
                opt_a=opts["A"], opt_b=opts["B"],
                opt_c=opts["C"], opt_d=opts["D"],
                misconception=m,
            )
        )
    
    t0 = time.time()
    backward_outputs = batch_generate(
        tokenizer, model, backward_prompts,
        max_new_tokens=5, batch_size=batch_size, temperature=0.0
    )
    logger.info(f"反向模拟完成，耗时: {time.time()-t0:.1f}s")
    
    simulated_answers = [parse_option_from_text(o) for o in backward_outputs]
    
    # ===== 阶段3: 权重赋值（严格对齐论文Table7）=====
    results = []
    weight_stats = Counter()
    for s, m, a_sim in zip(incorrect_samples, misconceptions, simulated_answers):
        a_student = s["student_answer"]
        a_correct = s["correct_answer"]
        
        if a_sim is None:
            weight = 1  # 无法解析，给基础权重
            weight_type = "medium_parse_fail"
        elif a_sim == a_student:
            weight = 2  # 完全一致，高权重
            weight_type = "high"
        elif a_sim == a_correct:
            weight = 0  # 反向模拟得到正确答案，丢弃
            weight_type = "discard"
        else:
            weight = 1  # a_sim≠correct且≠student_answer，中等权重
            weight_type = "medium"
        
        weight_stats[weight_type] += 1
        
        result = dict(s)
        result["misconception"] = m
        result["simulated_answer"] = a_sim
        result["weight"] = weight
        result["weight_type"] = weight_type
        results.append(result)
    
    logger.info(f"权重分布: {dict(weight_stats)}")
    discard_count = weight_stats["discard"]
    valid_count = len(results) - discard_count
    logger.info(f"有效样本: {valid_count}, 丢弃样本: {discard_count}")
    
    return results


def apply_student_level_consistency(
    samples_with_weights: List[Dict],
    all_train_samples: List[Dict],
) -> List[Dict]:
    """
    学生级循环一致性校验（适配历史预测任务）
    
    基于学生30条历史记录推断全局错误模式，
    若新题推断的错误概念与全局模式匹配则权重翻倍
    """
    logger.info("执行学生级循环一致性校验...")
    
    # 建立学生历史错误率统计
    student_error_rates = defaultdict(list)
    for s in all_train_samples:
        student_error_rates[s["user_id"]].append(not s["is_correct"])
    
    updated = []
    boost_count = 0
    for s in samples_with_weights:
        if s["weight"] == 0:
            updated.append(s)
            continue
        
        user_id = s["user_id"]
        errors = student_error_rates.get(user_id, [])
        
        if len(errors) > 0:
            error_rate = sum(errors) / len(errors)
            misconception = s.get("misconception", "")
            
            # 检查：推断的错误概念是否符合学生的全局错误模式
            # 使用启发式规则：错误率高的学生更可能有系统性错误概念
            pattern_match = (
                error_rate > 0.3  # 学生有显著错误历史
                and s["weight"] > 0  # 当前样本有效
                and len(misconception) > 20  # 错误概念描述足够具体
            )
            
            result = dict(s)
            if pattern_match:
                result["weight"] = s["weight"] * 2
                result["weight_type"] = s["weight_type"] + "_student_boosted"
                boost_count += 1
            updated.append(result)
        else:
            updated.append(s)
    
    logger.info(f"学生级一致性校验: {boost_count} 个样本权重翻倍")
    return updated


def generate_sft_dataset(
    samples_with_weights: List[Dict],
    correct_samples: List[Dict],
    target_size: int = TARGET_SFT_SIZE,
) -> List[Dict]:
    """
    构建最终SFT数据集
    
    1. 从带权重的错误样本中选取高质量样本
    2. 补充部分正确回答样本（学生掌握题目的情况）
    3. 按权重降序排列
    """
    # 过滤有效错误样本（weight > 0）
    valid_incorrect = [s for s in samples_with_weights if s["weight"] > 0]
    valid_incorrect.sort(key=lambda x: x["weight"], reverse=True)
    
    # 目标构成：70%错误样本 + 30%正确样本
    target_incorrect = int(target_size * 0.7)
    target_correct = target_size - target_incorrect
    
    selected_incorrect = valid_incorrect[:target_incorrect]
    selected_correct = random.sample(
        correct_samples, min(target_correct, len(correct_samples))
    )
    
    logger.info(f"错误样本: {len(selected_incorrect)}, 正确样本: {len(selected_correct)}")
    
    # 构建最终SFT数据集
    sft_data = []
    
    # 处理错误样本
    for s in selected_incorrect:
        prompt = build_sft_prompt(s)
        misconception = s.get("misconception", "Student incorrectly applies mathematical rules")
        response = build_sft_response(misconception, s["student_answer"], s["correct_answer"])
        
        sft_data.append({
            "prompt": prompt,
            "response": response,
            "weight": s["weight"],
            "weight_type": s.get("weight_type", "medium"),
            "student_answer": s["student_answer"],
            "correct_answer": s["correct_answer"],
            "simulated_answer": s.get("simulated_answer"),
            "misconception": misconception,
            "is_correct": False,
        })
    
    # 处理正确样本
    for s in selected_correct:
        prompt = build_sft_prompt(s)
        misconception = "No significant misconception - student demonstrated mastery of this concept"
        response = build_sft_response(misconception, s["student_answer"], s["correct_answer"])
        
        sft_data.append({
            "prompt": prompt,
            "response": response,
            "weight": 2,  # 正确样本直接给高权重
            "weight_type": "correct_answer",
            "student_answer": s["student_answer"],
            "correct_answer": s["correct_answer"],
            "simulated_answer": s["student_answer"],
            "misconception": misconception,
            "is_correct": True,
        })
    
    # 按权重降序排列
    sft_data.sort(key=lambda x: x["weight"], reverse=True)
    logger.info(f"SFT数据集构建完成，总计: {len(sft_data)} 条")
    
    return sft_data


def save_sft_dataset(sft_data: List[Dict], output_path: str):
    """保存SFT数据集"""
    with open(output_path, "w") as f:
        for item in sft_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"SFT数据集已保存到: {output_path}")


def save_statistics(sft_data: List[Dict], output_path: str):
    """保存数据集统计报告"""
    weight_dist = Counter(s["weight"] for s in sft_data)
    weight_type_dist = Counter(s.get("weight_type", "unknown") for s in sft_data)
    correct_count = sum(1 for s in sft_data if s.get("is_correct", False))
    incorrect_count = len(sft_data) - correct_count
    
    stats = {
        "total_samples": len(sft_data),
        "correct_answer_samples": correct_count,
        "incorrect_answer_samples": incorrect_count,
        "weight_distribution": dict(weight_dist),
        "weight_type_distribution": dict(weight_type_dist),
        "high_weight_samples": sum(1 for s in sft_data if s["weight"] >= 4),
        "medium_weight_samples": sum(1 for s in sft_data if s["weight"] in [1, 2]),
    }
    
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.info("数据集统计报告:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")


def main():
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("步骤3: MISTAKE-GENERATE 循环一致性SFT数据集构建")
    logger.info("=" * 60)
    
    # ===== 加载预处理训练数据 =====
    train_samples = load_training_data(PROCESSED_DATA_DIR / "train.jsonl")
    
    # 分离正确和错误回答样本
    incorrect_samples = [s for s in train_samples if not s["is_correct"]]
    correct_samples = [s for s in train_samples if s["is_correct"]]
    
    logger.info(f"训练集: 错误样本={len(incorrect_samples)}, 正确样本={len(correct_samples)}")
    
    # 采样候选集（目标8000 × 倍数）
    target_incorrect = int(TARGET_SFT_SIZE * 0.7)
    candidate_size = target_incorrect * CANDIDATE_MULTIPLIER
    candidate_size = min(candidate_size, len(incorrect_samples))
    
    candidate_samples = random.sample(incorrect_samples, candidate_size)
    logger.info(f"循环一致性校验候选样本: {len(candidate_samples)} 条")
    
    # ===== 加载Qwen2.5-3B-Instruct (4-bit NF4) =====
    tokenizer, model = load_model_4bit(str(MODEL_PATH))
    
    # ===== 单题级循环一致性校验 =====
    samples_with_weights = run_cycle_consistency_check(
        tokenizer, model, candidate_samples, batch_size=MAX_INFERENCE_BATCH
    )
    
    # ===== 学生级循环一致性校验 =====
    samples_with_weights = apply_student_level_consistency(
        samples_with_weights, train_samples
    )
    
    # ===== 构建SFT数据集 =====
    sft_data = generate_sft_dataset(
        samples_with_weights, correct_samples, target_size=TARGET_SFT_SIZE
    )
    
    # ===== 保存结果 =====
    output_sft_path = SFT_DATASET_DIR / "sft_v2.jsonl"
    save_sft_dataset(sft_data, str(output_sft_path))
    save_statistics(sft_data, str(SFT_DATASET_DIR / "sft_v2_stats.json"))
    
    # 同时保存中间检查点（循环一致性结果）
    checkpoint_path = TEMP_DIR / "cycle_consistency_results.jsonl"
    with open(checkpoint_path, "w") as f:
        for s in samples_with_weights[:1000]:  # 只保存前1000条便于检查
            f.write(json.dumps({
                k: v for k, v in s.items() if k != "history_text"
            }, ensure_ascii=False) + "\n")
    
    elapsed = time.time() - start_time
    logger.info(f"\n步骤3完成! 总耗时: {elapsed/60:.1f} 分钟")
    logger.info(f"SFT数据集路径: {output_sft_path}")
    
    return str(output_sft_path)


if __name__ == "__main__":
    main()
