import pandas as pd
import json
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from openai import AsyncOpenAI
import asyncio
import matplotlib.pyplot as plt
import seaborn as sns
import time

# ==========================================
# 配置参数
# ==========================================
DEEPSEEK_API_KEY = "sk-7b605cf119714ec4b429db287fd51073" # 请替换为您的 DeepSeek API Key
INPUT_DATA_FILE = "top20_users_records.jsonl"
CONCURRENCY_LIMIT = 20  # 最大并发请求数，根据您的 API 限流情况可调大或调小

client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# 解决图表中文显示问题，防止负号显示为方块
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 步骤 1 & 2: 基础数据处理与 IRT 估计
# ==========================================
def load_and_estimate_1pl(file_path):
    print("1. 开始读取数据并训练初始 1PL 模型...")
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    df = pd.DataFrame(data)
    df['score'] = df['IsCorrect'].map({'正确': 1, '错误': 0})
    df['UserId_str'] = 'U_' + df['UserId'].astype(str)
    df['question_id_str'] = 'Q_' + df['question_id'].astype(str)
    
    X_users = pd.get_dummies(df['UserId_str'])
    X_items = pd.get_dummies(df['question_id_str'])
    X = sp.hstack([sp.csr_matrix(X_users.values), sp.csr_matrix(X_items.values)])
    y = df['score'].values
    
    model = LogisticRegression(fit_intercept=False, C=1.0, solver='lbfgs', max_iter=2000)
    model.fit(X, y)
    
    user_cols = X_users.columns
    item_cols = X_items.columns
    theta_estimates = model.coef_[0][:len(user_cols)]
    b_estimates = -model.coef_[0][len(user_cols):]
    
    user_df = pd.DataFrame({'UserId': [c.replace('U_', '') for c in user_cols], 'theta': theta_estimates})
    item_df = pd.DataFrame({'question_id': [c.replace('Q_', '') for c in item_cols], 'b': b_estimates})
    
    print("-> 初始 IRT 估计完成。")
    return df, user_df, item_df, X_users, X_items

# ==========================================
# 步骤 3: 异步构建未作答题目 & 获取历史上下文
# ==========================================
def get_student_history(df, user_id, sample_size=2):
    user_records = df[df['UserId'].astype(str) == str(user_id)]
    correct_records = user_records[user_records['score'] == 1]
    wrong_records = user_records[user_records['score'] == 0]
    
    correct_samples = correct_records.sample(min(len(correct_records), sample_size)).to_dict('records')
    wrong_samples = wrong_records.sample(min(len(wrong_records), sample_size)).to_dict('records')
    return correct_samples, wrong_samples

async def simulate_single_answer(semaphore, user_id, item_id, theta, target_question, correct_samples, wrong_samples):
    """异步调用单一题目的模拟作答"""
    ability_desc = "普通水平"
    if theta > 1.0: ability_desc = "优秀水平"
    elif theta < -1.0: ability_desc = "较差水平"

    system_prompt = f"你现在正在扮演一名能力为【{ability_desc}】的学生进行答题测试。你需要根据你已有的知识水平来回答问题。请直接给出最终选项答案，不要过多解释。"
    
    user_prompt = "这是你过去做对的题目示例：\n"
    for idx, q in enumerate(correct_samples):
        user_prompt += f"题目{idx+1}: {q.get('question_text', '缺失题干')}\n你的回答是对的。\n"
        
    user_prompt += "\n这是你过去做错的题目示例：\n"
    for idx, q in enumerate(wrong_samples):
        user_prompt += f"题目{idx+1}: {q.get('question_text', '缺失题干')}\n这道题你做错了。\n"

    target_q_text = target_question.get('question_text', '缺失题干')
    target_q_options = target_question.get('options', '')
    
    user_prompt += f"\n现在，请回答下面这道新题目：\n题目: {target_q_text}\n选项: {target_q_options}\n\n请直接回复你的最终答案："

    async with semaphore: # 使用信号量限制并发数
        try:
            response = await client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=50
            )
            answer = response.choices[0].message.content.strip()
            
            real_answer = target_question.get('answer', '')
            is_correct = 1 if (real_answer and real_answer in answer) else 0
            
            return {
                "UserId": user_id,
                "question_id": item_id,
                "IsCorrect": "正确" if is_correct else "错误",
                "score": is_correct,
                "is_simulated": True
            }
        except Exception as e:
            print(f"API 调用失败 (User: {user_id}, Item: {item_id}): {e}")
            return None

async def run_all_simulations(df_real, user_df, questions_dict):
    """并发运行所有未作答题目"""
    all_users = df_real['UserId'].unique()
    all_items = df_real['question_id'].unique()
    
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = []
    
    print("2. 开始异步并行调用 DeepSeek 进行 SMART 模拟作答 (全量运行)...")
    for user_id in all_users:
        theta = user_df[user_df['UserId'] == str(user_id)]['theta'].values[0]
        attempted_items = df_real[df_real['UserId'] == user_id]['question_id'].values
        unattempted_items = [item for item in all_items if item not in attempted_items]
        
        correct_samples, wrong_samples = get_student_history(df_real, user_id)
        
        for item_id in unattempted_items:
            target_question = questions_dict.get(item_id, {})
            # 将所有任务加入事件循环池
            task = asyncio.create_task(
                simulate_single_answer(semaphore, user_id, item_id, theta, target_question, correct_samples, wrong_samples)
            )
            tasks.append(task)
    
    # 等待所有 API 请求并发完成
    print(f"共生成 {len(tasks)} 个模拟作答任务，正在火速请求中...")
    results = await asyncio.gather(*tasks)
    
    # 过滤掉请求失败的 None 结果
    simulated_records = [res for res in results if res is not None]
    print(f"-> 模拟作答完成，成功获取 {len(simulated_records)} 条预测记录。")
    return simulated_records

# ==========================================
# 步骤 4: 可视化展示函数
# ==========================================
def plot_results(user_df, item_df, item_df_new, comparison_df):
    print("4. 正在生成系列可视化图表...")
    sns.set(style="whitegrid") # 设回seaborn默认以防被覆盖
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    
    # --- 图 1: SMART 难度对齐分布 ---
    plt.figure(figsize=(8, 6))
    sns.kdeplot(item_df['b'], fill=True, label='初始 1PL 难度 (仅真实数据)', color='skyblue')
    sns.kdeplot(item_df_new['b_combined'], fill=True, label='SMART 对齐后难度 (真实+模拟)', color='salmon')
    plt.title('题目难度分布对比', fontsize=15)
    plt.xlabel('题目难度 (b)', fontsize=12)
    plt.ylabel('密度', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('SMART_IRT_Comparison.png', dpi=300)
    plt.show()

    # --- 图 2：可视化题目难度分布 ---
    plt.figure(figsize=(10, 6))
    sns.histplot(comparison_df['b_new'], bins=30, kde=True, color='salmon')
    plt.title('Distribution of Item Difficulty (b)', fontsize=15)
    plt.xlabel('Difficulty Level (b)', fontsize=12)
    plt.ylabel('Count of Items', fontsize=12)
    plt.tight_layout()
    plt.savefig('item_difficulty_dist_smart.png')
    plt.show()

    # --- 图 3：可视化项目特征曲线 (ICC) ---
    item_df_sorted = comparison_df.sort_values('b_new').reset_index(drop=True)
    num_items = len(item_df_sorted)

    sample_indices = [0, num_items//4, num_items//2, 3*num_items//4, num_items-1]
    sample_items = item_df_sorted.iloc[sample_indices]

    def icc_prob(theta, b):
        return 1 / (1 + np.exp(-(theta - b)))

    theta_range = np.linspace(-4, 4, 100)

    plt.figure(figsize=(12, 8))
    colors = sns.color_palette("husl", len(sample_items))

    for i, (idx, row) in enumerate(sample_items.iterrows()):
        b = row['b_new']
        q_id = row['question_id']
        probs = icc_prob(theta_range, b)
        plt.plot(theta_range, probs, label=f'Question {q_id} (b={b:.2f})', color=colors[i], linewidth=2.5)

    plt.title('Item Characteristic Curves (ICC) for Selected Questions', fontsize=16)
    plt.xlabel('User Ability Level (Theta)', fontsize=14)
    plt.ylabel('Probability of Correct Answer P(X=1)', fontsize=14)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5) 
    plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5) 
    plt.legend(title='Items', loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('icc_plot.png')
    plt.show()

    print("-> 所有可视化图表已生成并保存。")

# ==========================================
# 主流程执行
# ==========================================
async def main():
    start_time = time.time()
    
    # 1. 初始估计
    df_real, user_df, item_df, X_users, X_items = load_and_estimate_1pl(INPUT_DATA_FILE)
    
    # 获取题目详情字典 (包含选项和答案)
    questions_dict = df_real.drop_duplicates('question_id').set_index('question_id').to_dict('index')

    # 2. 并发进行模拟作答
    simulated_records = await run_all_simulations(df_real, user_df, questions_dict)

    # 3. 结合模拟数据，进行整体 IRT 测试
    print("3. 合并真实数据与模拟数据，重新进行 IRT 评估...")
    df_sim = pd.DataFrame(simulated_records)
    if not df_sim.empty:
        df_sim['UserId_str'] = 'U_' + df_sim['UserId'].astype(str)
        df_sim['question_id_str'] = 'Q_' + df_sim['question_id'].astype(str)
        
        df_real['is_simulated'] = False
        df_combined = pd.concat([df_real, df_sim], ignore_index=True)
    else:
        print("警告：模拟数据为空，将只使用真实数据进行后续计算。")
        df_real['is_simulated'] = False
        df_combined = df_real
    
    X_users_new = pd.get_dummies(df_combined['UserId_str'])
    X_items_new = pd.get_dummies(df_combined['question_id_str'])
    X_new = sp.hstack([sp.csr_matrix(X_users_new.values), sp.csr_matrix(X_items_new.values)])
    y_new = df_combined['score'].values
    
    model_combined = LogisticRegression(fit_intercept=False, C=1.0, solver='lbfgs', max_iter=2000)
    model_combined.fit(X_new, y_new)
    
    item_cols_new = X_items_new.columns
    b_estimates_new = -model_combined.coef_[0][len(X_users_new.columns):]
    
    item_df_new = pd.DataFrame({
        'question_id': [c.replace('Q_', '') for c in item_cols_new], 
        'b_combined': b_estimates_new
    })
    
    # 4. 合并数据、保存与可视化
    comparison_df = pd.merge(item_df.rename(columns={'b': 'b_old'}), 
                             item_df_new.rename(columns={'b_combined': 'b_new'}), 
                             on='question_id', how='inner')
    
    # 保存 CSV 结果供以后查询
    user_df.to_csv("user_ability.csv", index=False)
    comparison_df.to_csv("item_difficulty_smart_aligned.csv", index=False)
    print("-> 结果数据已保存至 'user_ability.csv' 和 'item_difficulty_smart_aligned.csv'")
    
    # 内存直传数据进行可视化，省去二次读取的时间
    plot_results(user_df, item_df, item_df_new, comparison_df)
    
    end_time = time.time()
    print(f"\n全部任务执行完毕！总耗时: {end_time - start_time:.2f} 秒。")

if __name__ == "__main__":
    # 使用 asyncio 运行主函数
    asyncio.run(main())