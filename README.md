# StuSim-Baseline

> 复现 MIT 论文 [**Learning to Make Mistakes**] 的 MISTAKE-CYCLE+CORRECT 方法

---

## 实验结果



| 指标 | 基座模型 | SFT 后 | 
|------|---------|--------|
| 整体预测准确率 | 24.69% | **34.77%** | 
| 学生正确子集准确率 | 22.95% | **52.87%** | 
| 学生错误子集准确率 | 26.45% | 16.53% | 
| 循环一致性通过率 | 25.50% | 19.50% |



>Knowledge Tracing baseline
## 实验结果
输入：时间序列（对学生）  

     每一步  （q_t,r_t)
     
      q_t :第t次做的题  question_id/skill_id       r_t：做对(1)还是做错(0)

预测：下一题做对还是做错（random 50%）

①按学生分组，给question_id 预测下一题选什么  准确率：54.37%

②按学生分组，给question_id 预测下一题做对还是做错 准确率：71.85%  AUC：78.42%

③按题号分组，给question_id 预测下一题做对还是做错 准确率：52.28%  AUC：52.85%


>IRT
## 实验结果
数据集中被试者的能力分布和题目难度发布均符合正态分布
<img width="1000" height="600" alt="user_ability_dist" src="https://github.com/user-attachments/assets/58625e67-0147-43ca-8578-b52a6acbd334" />
<img width="1000" height="600" alt="item_difficulty_dist" src="https://github.com/user-attachments/assets/738f8604-7e82-49b3-ab64-c704afbde27b" />



