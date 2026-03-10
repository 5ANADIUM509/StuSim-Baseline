#!/usr/bin/env python3
"""
步骤6: 收尾与文件整理
清理临时文件，整理归档，验证输出完整性
"""

import os, shutil, json, logging
from pathlib import Path
from datetime import datetime

# ========== 路径配置 ==========
BASE_DIR = Path("/root/autodl-tmp")
TEMP_DIR = BASE_DIR / "temp"
PIP_CACHE_DIR = BASE_DIR / "pip_cache"
LOG_DIR = BASE_DIR / "logs"
EVAL_REPORT_DIR = BASE_DIR / "evaluation_report"
TRAINED_MODEL_DIR = BASE_DIR / "trained_model"
SFT_DATASET_DIR = BASE_DIR / "sft_dataset"
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"

# ========== 日志配置 ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "step6_cleanup.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def clean_temp_files():
    """清理临时文件（保留推理缓存和重要中间结果）"""
    logger.info("清理临时文件...")
    
    if TEMP_DIR.exists():
        # 只删除不重要的临时文件，保留推理缓存
        keep_patterns = ["*_results.jsonl", "eval_samples.jsonl"]
        
        total_freed = 0
        for f in TEMP_DIR.iterdir():
            # 检查是否应该保留
            should_keep = any(f.match(p) for p in keep_patterns)
            if not should_keep:
                if f.is_file():
                    size = f.stat().st_size
                    f.unlink()
                    total_freed += size
                    logger.info(f"  删除: {f.name} ({size/1024:.1f}KB)")
                elif f.is_dir():
                    size = sum(ff.stat().st_size for ff in f.rglob("*") if ff.is_file())
                    shutil.rmtree(f)
                    total_freed += size
                    logger.info(f"  删除目录: {f.name} ({size/1024/1024:.1f}MB)")
        
        logger.info(f"临时文件清理完成，释放: {total_freed/1024/1024:.1f}MB")
    
    # 清理pip缓存中的旧包（只保留最近使用的）
    if PIP_CACHE_DIR.exists():
        http_cache = PIP_CACHE_DIR / "http"
        if http_cache.exists():
            cache_size = sum(f.stat().st_size for f in http_cache.rglob("*") if f.is_file())
            if cache_size > 500 * 1024 * 1024:  # 超过500MB才清理
                shutil.rmtree(http_cache)
                logger.info(f"清理pip HTTP缓存: {cache_size/1024/1024:.1f}MB")
    
    # 清理系统盘缓存（~/.cache下的huggingface/pip缓存）
    home_hf_cache = Path.home() / ".cache/huggingface"
    if home_hf_cache.exists():
        size = sum(f.stat().st_size for f in home_hf_cache.rglob("*") if f.is_file())
        logger.info(f"检测到系统盘HuggingFace缓存: {size/1024/1024:.1f}MB")
        if size > 10 * 1024 * 1024:  # 超过10MB才清理
            shutil.rmtree(home_hf_cache)
            logger.info("已清理系统盘HuggingFace缓存")
    
    home_pip_cache = Path.home() / ".cache/pip"
    if home_pip_cache.exists():
        size = sum(f.stat().st_size for f in home_pip_cache.rglob("*") if f.is_file())
        logger.info(f"检测到系统盘pip缓存: {size/1024/1024:.1f}MB")
        if size > 10 * 1024 * 1024:
            shutil.rmtree(home_pip_cache)
            logger.info("已清理系统盘pip缓存")


def create_inference_script():
    """创建一键推理脚本"""
    script_content = '''#!/usr/bin/env python3
"""
一键推理脚本
用法: python infer.py --history_file history.jsonl --question "..." --opt_a "..." --opt_b "..." --opt_c "..." --opt_d "..."
"""

import os, json, re, argparse
from pathlib import Path

os.environ["HF_HOME"] = "/root/autodl-tmp/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "/root/autodl-tmp/huggingface_cache"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "/root/autodl-tmp/models/Qwen/Qwen2.5-3B-Instruct"
LORA_MODEL = "/root/autodl-tmp/trained_model/best_model_v2"

SYSTEM_PROMPT = """You are an expert mathematics educator specializing in student misconception analysis.

CRITICAL CONSTRAINT: Any inferred student misconception must be restorable - 
using that misconception to simulate solving must produce the student\'s actual chosen answer."""

USER_PROMPT = """Based on this student\'s 30-question history, identify their misconception pattern and predict their answer.

Student History:
{history_text}

New Question: {new_question}
A. {new_opt_a}
B. {new_opt_b}
C. {new_opt_c}
D. {new_opt_d}

Analyze the student\'s recurring error patterns from their history, then predict:
1. The main misconception driving their errors
2. Which option they will MOST LIKELY choose (based on their history patterns, not what\'s correct)
3. Which option is actually correct

Output ONLY valid JSON:
{{"misconception_pattern": "one sentence describing the main pattern", "predicted_student_option": "X", "predicted_correct_option": "Y"}}"""


def load_history(history_file: str, n: int = 30) -> str:
    """加载最近n条历史记录"""
    records = []
    with open(history_file) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    
    records = records[-n:]  # 取最后n条
    while len(records) < n:
        records.insert(0, records[0] if records else {"question": "N/A", "student_ans": "N/A", "correct_ans": "N/A", "is_correct": True})
    
    lines = []
    for i, r in enumerate(records, 1):
        mark = "✓" if r.get("is_correct", True) else "✗"
        lines.append(
            f"{i}. 问题: {r.get(\'question\', \'N/A\')} | "
            f"学生答案: {r.get(\'student_ans\', \'N/A\')} ({mark}) | "
            f"正确答案: {r.get(\'correct_ans\', \'N/A\')}"
        )
    return "\\n".join(lines)


def predict(history_text, question, opt_a, opt_b, opt_c, opt_d):
    """执行预测"""
    print("加载模型...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )
    
    if Path(LORA_MODEL).exists():
        model = PeftModel.from_pretrained(model, LORA_MODEL)
        print("LoRA微调模型加载成功")
    
    model.eval()
    
    user_content = USER_PROMPT.format(
        history_text=history_text, new_question=question,
        new_opt_a=opt_a, new_opt_b=opt_b, new_opt_c=opt_c, new_opt_d=opt_d,
    )
    prompt = f"<|im_start|>system\\n{SYSTEM_PROMPT}<|im_end|>\\n<|im_start|>user\\n{user_content}<|im_end|>\\n<|im_start|>assistant\\n"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=100, do_sample=False,
            temperature=1.0, pad_token_id=tokenizer.eos_token_id,
        )
    
    input_len = inputs["input_ids"].shape[1]
    generated = tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip()
    
    print("\\n模型输出:", generated)
    
    try:
        match = re.search(r\'\\{[^{}]*\\}\', generated, re.DOTALL)
        if match:
            result = json.loads(match.group())
            print("\\n预测结果:")
            print(f"  学生可能选择: {result.get(\'predicted_student_option\', \'N/A\')}")
            print(f"  正确答案是: {result.get(\'predicted_correct_option\', \'N/A\')}")
            print(f"  错误概念: {result.get(\'misconception_pattern\', \'N/A\')}")
            return result
    except Exception as e:
        print(f"解析失败: {e}")
    
    return {"raw": generated}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MISTAKE模型一键推理")
    parser.add_argument("--history_file", type=str, help="历史记录JSONL文件")
    parser.add_argument("--history_text", type=str, help="直接提供历史记录文本")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--opt_a", type=str, required=True)
    parser.add_argument("--opt_b", type=str, required=True)
    parser.add_argument("--opt_c", type=str, required=True)
    parser.add_argument("--opt_d", type=str, required=True)
    
    args = parser.parse_args()
    
    if args.history_file:
        history_text = load_history(args.history_file)
    elif args.history_text:
        history_text = args.history_text
    else:
        raise ValueError("必须提供 --history_file 或 --history_text")
    
    predict(history_text, args.question, args.opt_a, args.opt_b, args.opt_c, args.opt_d)
'''
    
    script_path = BASE_DIR / "infer.py"
    with open(script_path, "w") as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    logger.info(f"一键推理脚本已创建: {script_path}")


def generate_file_manifest():
    """生成所有输出文件清单"""
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "output_files": [],
    }
    
    check_dirs = [
        PROCESSED_DATA_DIR,
        SFT_DATASET_DIR,
        TRAINED_MODEL_DIR,
        EVAL_REPORT_DIR,
        LOG_DIR,
    ]
    
    for dir_path in check_dirs:
        if dir_path.exists():
            for f in dir_path.rglob("*"):
                if f.is_file():
                    manifest["output_files"].append({
                        "path": str(f.relative_to(BASE_DIR)),
                        "size_mb": round(f.stat().st_size / 1024 / 1024, 2),
                        "category": dir_path.name,
                    })
    
    manifest_path = BASE_DIR / "output_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    logger.info(f"文件清单已保存: {manifest_path}")
    logger.info(f"总输出文件数: {len(manifest['output_files'])}")
    
    # 统计各类文件大小
    from collections import defaultdict
    size_by_category = defaultdict(float)
    for f in manifest["output_files"]:
        size_by_category[f["category"]] += f["size_mb"]
    
    logger.info("各类文件大小:")
    for cat, size in size_by_category.items():
        logger.info(f"  {cat}: {size:.1f}MB")
    
    total_size = sum(f["size_mb"] for f in manifest["output_files"])
    logger.info(f"总大小: {total_size:.1f}MB")


def verify_outputs():
    """验证所有关键输出文件是否存在"""
    required_files = [
        PROCESSED_DATA_DIR / "train.jsonl",
        PROCESSED_DATA_DIR / "val.jsonl",
        PROCESSED_DATA_DIR / "test.jsonl",
        EVAL_REPORT_DIR / "evaluation_report_v2.md",
        EVAL_REPORT_DIR / "metrics_v2.json",
    ]
    
    optional_files = [
        SFT_DATASET_DIR / "sft_v2.jsonl",
        TRAINED_MODEL_DIR / "best_model_v2",
        BASE_DIR / "infer.py",
    ]
    
    logger.info("验证输出文件完整性...")
    all_ok = True
    
    for f in required_files:
        if f.exists():
            logger.info(f"  ✅ {f.relative_to(BASE_DIR)}")
        else:
            logger.warning(f"  ❌ 缺失: {f.relative_to(BASE_DIR)}")
            all_ok = False
    
    for f in optional_files:
        if f.exists():
            logger.info(f"  ✅ {f.relative_to(BASE_DIR)} (可选)")
        else:
            logger.warning(f"  ⚠️ {f.relative_to(BASE_DIR)} (可选，未找到)")
    
    return all_ok


def main():
    logger.info("=" * 60)
    logger.info("步骤6: 收尾与文件整理")
    logger.info("=" * 60)
    
    # 清理临时文件
    clean_temp_files()
    
    # 创建一键推理脚本
    create_inference_script()
    
    # 生成文件清单
    generate_file_manifest()
    
    # 验证输出完整性
    all_ok = verify_outputs()
    
    if all_ok:
        logger.info("\n✅ 全流程完成！所有关键输出文件验证通过")
    else:
        logger.warning("\n⚠️ 部分关键文件缺失，请检查上述日志")
    
    logger.info("步骤6完成")


if __name__ == "__main__":
    main()
