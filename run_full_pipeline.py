#!/usr/bin/env python3
"""
MISTAKE全流程主控制脚本
基于论文《LEARNING TO MAKE MISTAKES》的MISTAKE-CYCLE+CORRECT变体

执行顺序:
  步骤1: 环境初始化与验证
  步骤2: 数据预处理确认
  步骤3: MISTAKE-GENERATE 循环一致性SFT数据集构建 (1.5h)
  步骤4: MISTAKE-UPDATE 迭代式LoRA微调 (2h)
  步骤5: 全维度效果评估 (30min)
  步骤6: 收尾整理

总时长控制: ≤5小时
磁盘写入: 100%写入 /root/autodl-tmp/
"""

import os, sys, json, logging, time, subprocess
from pathlib import Path
from datetime import datetime

# ==================== 强制数据盘路径（必须最先设置）====================
BASE_DIR = Path("/root/autodl-tmp")
TEMP_DIR = BASE_DIR / "temp"
LOG_DIR = BASE_DIR / "logs"
HF_CACHE = BASE_DIR / "huggingface_cache"
TORCH_CACHE = BASE_DIR / "torch_cache"
PIP_CACHE = BASE_DIR / "pip_cache"

# 确保所有目录存在
for d in [TEMP_DIR, LOG_DIR, HF_CACHE, TORCH_CACHE, PIP_CACHE]:
    d.mkdir(parents=True, exist_ok=True)

# 设置环境变量
os.environ.update({
    "HF_HOME": str(HF_CACHE),
    "TRANSFORMERS_CACHE": str(HF_CACHE),
    "TORCH_HOME": str(TORCH_CACHE),
    "HF_DATASETS_CACHE": str(HF_CACHE / "datasets"),
    "PIP_CACHE_DIR": str(PIP_CACHE),
    "TMPDIR": str(TEMP_DIR),
    "TRANSFORMERS_OFFLINE": "1",  # 禁止在线下载（使用本地模型）
    "HF_DATASETS_OFFLINE": "1",
    "PYTHONHASHSEED": "42",
})

# 验证系统盘写入保护
def verify_no_system_disk_write():
    """验证关键环境变量指向数据盘"""
    checks = {
        "HF_HOME": os.environ.get("HF_HOME", ""),
        "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE", ""),
        "TORCH_HOME": os.environ.get("TORCH_HOME", ""),
    }
    for key, val in checks.items():
        if not val.startswith("/root/autodl-tmp"):
            print(f"❌ 错误: {key}={val} 未指向数据盘！")
            sys.exit(1)
    return True

verify_no_system_disk_write()

# ==================== 日志配置 ====================
LOG_FILE = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ==================== 时间控制 ====================
PIPELINE_START = time.time()
TIME_LIMITS = {
    "step3": 90 * 60,   # 1.5小时
    "step4": 120 * 60,  # 2小时
    "step5": 30 * 60,   # 30分钟
    "step6": 10 * 60,   # 10分钟
    "total": 300 * 60,  # 5小时总限制
}


def check_time_limit(step_name: str) -> bool:
    """检查是否还在时间限制内"""
    elapsed = time.time() - PIPELINE_START
    remaining = TIME_LIMITS["total"] - elapsed
    
    if remaining < 60:
        logger.warning(f"⚠️ 时间不足1分钟，跳过步骤: {step_name}")
        return False
    
    step_limit = TIME_LIMITS.get(step_name, 30 * 60)
    if remaining < step_limit / 2:
        logger.warning(f"⚠️ 剩余时间 {remaining/60:.1f}min 不足以完成 {step_name}，将限时运行")
    
    return True


def elapsed_str():
    """返回已用时间字符串"""
    elapsed = time.time() - PIPELINE_START
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ==================== 步骤1: 环境初始化 ====================
def step1_env_init():
    """环境初始化与依赖检查"""
    logger.info("=" * 70)
    logger.info("步骤1: 环境初始化与验证")
    logger.info("=" * 70)
    
    # 验证环境变量
    verify_no_system_disk_write()
    logger.info("✅ 数据盘路径配置验证通过")
    
    # 验证GPU
    try:
        import torch
        assert torch.cuda.is_available(), "CUDA不可用"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"✅ GPU: {gpu_name}, 显存: {gpu_mem:.1f}GB")
    except Exception as e:
        logger.error(f"❌ GPU检查失败: {e}")
        sys.exit(1)
    
    # 验证关键依赖
    required_packages = {
        "transformers": "4.40",
        "peft": "0.7",
        "bitsandbytes": "0.40",
        "trl": "0.8",
    }
    
    for pkg, min_ver in required_packages.items():
        try:
            import importlib
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "unknown")
            logger.info(f"  ✅ {pkg}=={ver}")
        except ImportError:
            logger.warning(f"  ⚠️ {pkg} 未安装，尝试安装...")
            os.system(f"pip install {pkg} --cache-dir {PIP_CACHE} -q")
    
    # 验证模型文件
    model_path = BASE_DIR / "models/Qwen/Qwen2.5-3B-Instruct"
    if not model_path.exists():
        # 尝试备用路径
        alt_path = BASE_DIR / "models/Qwen/Qwen2___5-3B-Instruct"
        if alt_path.exists():
            logger.warning(f"使用备用模型路径: {alt_path}")
        else:
            logger.error(f"❌ 模型文件不存在: {model_path}")
            sys.exit(1)
    else:
        logger.info(f"✅ 基座模型已存在: {model_path}")
    
    # 验证数据集
    dataset_path = BASE_DIR / "train_dataset.filtered.jsonl"
    if not dataset_path.exists():
        logger.error(f"❌ 数据集不存在: {dataset_path}")
        sys.exit(1)
    else:
        logger.info(f"✅ 原始数据集已存在")
    
    # 固定随机种子
    import random, numpy as np
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    logger.info("✅ 随机种子42已固定")
    
    logger.info(f"步骤1完成 [{elapsed_str()}]")
    return True


# ==================== 步骤2: 数据预处理确认 ====================
def step2_data_preprocessing():
    """
    确认数据预处理状态
    如果预处理数据已存在（由之前的运行生成），直接复用
    否则执行预处理
    """
    logger.info("=" * 70)
    logger.info("步骤2: 数据预处理确认")
    logger.info("=" * 70)
    
    processed_dir = BASE_DIR / "processed_data"
    required_files = ["train.jsonl", "val.jsonl", "test.jsonl"]
    
    all_exist = all((processed_dir / f).exists() for f in required_files)
    
    if all_exist:
        logger.info("✅ 预处理数据已存在，直接复用")
        
        # 打印统计信息
        stats_path = processed_dir / "stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            for k, v in stats.items():
                logger.info(f"  {k}: {v}")
        
        # 确认test_correct和test_incorrect子集存在
        for subset in ["test_correct.jsonl", "test_incorrect.jsonl"]:
            subset_path = processed_dir / subset
            if not subset_path.exists():
                logger.info(f"创建测试集子集: {subset}")
                _create_test_subsets(processed_dir)
                break
        
        logger.info(f"步骤2完成 [{elapsed_str()}]")
        return True
    
    logger.info("预处理数据不存在，执行预处理...")
    return _run_preprocessing(processed_dir)


def _create_test_subsets(processed_dir: Path):
    """创建测试集正确/错误子集"""
    import json
    
    correct_samples = []
    incorrect_samples = []
    
    with open(processed_dir / "test.jsonl") as f:
        for line in f:
            s = json.loads(line)
            if s["is_correct"]:
                correct_samples.append(s)
            else:
                incorrect_samples.append(s)
    
    with open(processed_dir / "test_correct.jsonl", "w") as f:
        for s in correct_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    
    with open(processed_dir / "test_incorrect.jsonl", "w") as f:
        for s in incorrect_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    
    logger.info(f"测试集子集: 正确={len(correct_samples)}, 错误={len(incorrect_samples)}")


def _run_preprocessing(processed_dir: Path) -> bool:
    """执行数据预处理"""
    # 这里应该调用data_preprocessing.py或data_preprocessing_v2.py
    preprocess_scripts = [
        BASE_DIR / "data_preprocessing_v2.py",
        BASE_DIR / "data_preprocessing.py",
    ]
    
    for script in preprocess_scripts:
        if script.exists():
            logger.info(f"运行预处理脚本: {script}")
            result = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True, text=True, timeout=1800
            )
            if result.returncode == 0:
                logger.info("预处理完成")
                return True
            else:
                logger.error(f"预处理失败: {result.stderr[-500:]}")
    
    logger.error("所有预处理脚本均失败")
    return False


# ==================== 步骤3: MISTAKE-GENERATE ====================
def step3_mistake_generate():
    """执行MISTAKE-GENERATE循环一致性SFT数据集构建"""
    logger.info("=" * 70)
    logger.info("步骤3: MISTAKE-GENERATE 循环一致性SFT数据集构建")
    logger.info("=" * 70)
    
    if not check_time_limit("step3"):
        return None
    
    # 检查是否已有高质量SFT数据
    sft_v2_path = BASE_DIR / "sft_dataset/sft_v2.jsonl"
    if sft_v2_path.exists():
        logger.info(f"✅ SFT v2数据集已存在，跳过生成: {sft_v2_path}")
        with open(sft_v2_path) as f:
            count = sum(1 for _ in f)
        logger.info(f"  样本数: {count}")
        if count >= 5000:
            return str(sft_v2_path)
    
    t0 = time.time()
    
    try:
        sys.path.insert(0, str(BASE_DIR))
        from step3_mistake_generate import main as generate_main
        result = generate_main()
        
        elapsed = time.time() - t0
        logger.info(f"步骤3完成，耗时: {elapsed/60:.1f}分钟 [{elapsed_str()}]")
        return result
        
    except Exception as e:
        logger.error(f"步骤3失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # 回退到已有SFT数据
        fallback = BASE_DIR / "sft_dataset/sft_data.jsonl"
        if fallback.exists():
            logger.warning(f"使用备用SFT数据: {fallback}")
            return str(fallback)
        return None


# ==================== 步骤4: MISTAKE-UPDATE ====================
def step4_mistake_update():
    """执行MISTAKE-UPDATE迭代式LoRA微调"""
    logger.info("=" * 70)
    logger.info("步骤4: MISTAKE-UPDATE 迭代式LoRA微调")
    logger.info("=" * 70)
    
    if not check_time_limit("step4"):
        return None
    
    # 检查是否已有训练好的模型
    best_model = BASE_DIR / "trained_model/best_model_v2"
    if best_model.exists() and (best_model / "adapter_config.json").exists():
        logger.info(f"✅ 微调模型已存在，跳过训练: {best_model}")
        return str(best_model)
    
    t0 = time.time()
    
    try:
        sys.path.insert(0, str(BASE_DIR))
        from step4_mistake_update import main as update_main
        result = update_main()
        
        elapsed = time.time() - t0
        logger.info(f"步骤4完成，耗时: {elapsed/60:.1f}分钟 [{elapsed_str()}]")
        return result
        
    except Exception as e:
        logger.error(f"步骤4失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # 回退到已有模型
        fallback = BASE_DIR / "trained_model/best_model"
        if fallback.exists():
            logger.warning(f"使用备用模型: {fallback}")
            return str(fallback)
        return None


# ==================== 步骤5: 全维度评估 ====================
def step5_evaluate():
    """执行全维度效果评估"""
    logger.info("=" * 70)
    logger.info("步骤5: 全维度效果评估")
    logger.info("=" * 70)
    
    if not check_time_limit("step5"):
        logger.warning("时间不足，跳过完整评估，尝试快速评估...")
    
    t0 = time.time()
    
    try:
        sys.path.insert(0, str(BASE_DIR))
        from step5_evaluate import main as eval_main
        eval_main()
        
        elapsed = time.time() - t0
        logger.info(f"步骤5完成，耗时: {elapsed/60:.1f}分钟 [{elapsed_str()}]")
        return True
        
    except Exception as e:
        logger.error(f"步骤5失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


# ==================== 步骤6: 收尾整理 ====================
def step6_cleanup():
    """收尾与文件整理"""
    logger.info("=" * 70)
    logger.info("步骤6: 收尾与文件整理")
    logger.info("=" * 70)
    
    try:
        sys.path.insert(0, str(BASE_DIR))
        from step6_cleanup import main as cleanup_main
        cleanup_main()
        logger.info(f"步骤6完成 [{elapsed_str()}]")
        return True
    except Exception as e:
        logger.error(f"步骤6失败: {e}")
        return False


# ==================== 生成最终摘要报告 ====================
def generate_summary_report():
    """生成全流程执行摘要"""
    total_elapsed = time.time() - PIPELINE_START
    
    summary = {
        "pipeline_start": datetime.fromtimestamp(PIPELINE_START).isoformat(),
        "pipeline_end": datetime.now().isoformat(),
        "total_elapsed_minutes": round(total_elapsed / 60, 1),
        "status": "completed",
        "outputs": {
            "preprocessed_data": str(BASE_DIR / "processed_data"),
            "sft_dataset": str(BASE_DIR / "sft_dataset/sft_v2.jsonl"),
            "trained_model": str(BASE_DIR / "trained_model/best_model_v2"),
            "evaluation_report": str(BASE_DIR / "evaluation_report/evaluation_report_v2.md"),
            "inference_script": str(BASE_DIR / "infer.py"),
        },
    }
    
    # 读取评估结果
    metrics_path = BASE_DIR / "evaluation_report/metrics_v2.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        
        base = metrics.get("base_model", {})
        ft = metrics.get("fine_tuned_model", {})
        
        summary["evaluation_results"] = {
            "base_model_overall_accuracy": base.get("overall_accuracy"),
            "ft_model_overall_accuracy": ft.get("overall_accuracy"),
            "base_model_incorrect_subset_accuracy": base.get("incorrect_subset_accuracy"),
            "ft_model_incorrect_subset_accuracy": ft.get("incorrect_subset_accuracy"),
            "overall_improvement": (
                (ft.get("overall_accuracy", 0) - base.get("overall_accuracy", 0))
                if base.get("overall_accuracy") and ft.get("overall_accuracy") else None
            ),
        }
    
    summary_path = BASE_DIR / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info("\n" + "=" * 70)
    logger.info("全流程执行摘要")
    logger.info("=" * 70)
    logger.info(f"总耗时: {total_elapsed/60:.1f} 分钟")
    
    if "evaluation_results" in summary:
        er = summary["evaluation_results"]
        logger.info(f"\n核心评估结果:")
        logger.info(f"  整体准确率: {er.get('base_model_overall_accuracy', 0):.2%} → {er.get('ft_model_overall_accuracy', 0):.2%}")
        logger.info(f"  错误子集: {er.get('base_model_incorrect_subset_accuracy', 0):.2%} → {er.get('ft_model_incorrect_subset_accuracy', 0):.2%}")
        if er.get("overall_improvement"):
            logger.info(f"  整体提升: +{er['overall_improvement']:.2%}")
    
    return summary


# ==================== 主函数 ====================
def main():
    logger.info("=" * 70)
    logger.info("MISTAKE全流程主控制脚本")
    logger.info("论文: Learning to Make Mistakes (MIT)")
    logger.info("方法: MISTAKE-CYCLE+CORRECT 强约束变体")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    
    pipeline_results = {}
    
    # 步骤1: 环境初始化
    try:
        pipeline_results["step1"] = step1_env_init()
    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"步骤1致命错误: {e}")
        sys.exit(1)
    
    # 步骤2: 数据预处理
    pipeline_results["step2"] = step2_data_preprocessing()
    if not pipeline_results["step2"]:
        logger.error("步骤2失败，终止流程")
        sys.exit(1)
    
    # 步骤3: MISTAKE-GENERATE
    pipeline_results["step3"] = step3_mistake_generate()
    logger.info(f"步骤3状态: {'成功' if pipeline_results['step3'] else '跳过/失败'}")
    
    # 步骤4: MISTAKE-UPDATE
    pipeline_results["step4"] = step4_mistake_update()
    logger.info(f"步骤4状态: {'成功' if pipeline_results['step4'] else '跳过/失败'}")
    
    # 步骤5: 全维度评估
    pipeline_results["step5"] = step5_evaluate()
    logger.info(f"步骤5状态: {'成功' if pipeline_results['step5'] else '失败'}")
    
    # 步骤6: 收尾整理
    pipeline_results["step6"] = step6_cleanup()
    
    # 生成摘要
    summary = generate_summary_report()
    
    logger.info(f"\n日志文件: {LOG_FILE}")
    logger.info("全流程执行完毕！")


if __name__ == "__main__":
    main()
