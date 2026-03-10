"""
Prompt模板定义
"""

# =========================
# System Prompt
# =========================
SYSTEM_PROMPT = """You are an expert educator specializing in assessing mathematical understanding.

Your task is to analyze a student's past responses to diagnose their mathematical skills and predict how they will answer a new question."""

# =========================
# Default Prompt模板
# =========================
PROMPT_TEMPLATE = """Prompt:

You are given a history of 30 math questions, each including:
- Question
- Answer options
- Student's selected answer
- Correct answer

Student's History:

{history_text}

NEW PROBLEM
Question: {new_question}
Options:
A. {new_opt_a}
B. {new_opt_b}
C. {new_opt_c}
D. {new_opt_d}

Instructions:
1. For EACH historical question (1..30), write ONE likely misconception that explains the student's behavior on that item.
   - If the student answered correctly and confidently (or no clear misconception), write a plausible 'no misconception / mastered' statement grounded in the item.
   - If the student answered incorrectly, identify the most likely misconception that led to their wrong answer.
   - Pay special attention to patterns: which misconceptions appear multiple times? Which types of errors does the student make repeatedly?

2. For the NEW PROBLEM, identify the key skill(s) needed to solve it correctly.

3. For the NEW PROBLEM, for EACH option A/B/C/D, describe what misconception or reasoning pattern might lead a student to choose that option.

4. CRITICAL PREDICTION STEP - Analyze the student's historical patterns:
   a. Identify the most common misconceptions in the student's history (count how many times each type appears)
   b. Note which skills the student has mastered vs. struggles with
   c. Look for specific error patterns: Does the student consistently confuse certain concepts? (e.g., distributive vs. associative property)
   d. Check if the student has a pattern of choosing "Not enough information" or similar options when uncertain
   e. Match the new problem's required skills and potential misconceptions to the student's historical error patterns
   f. If the new problem requires skills the student has mastered, they are more likely to answer correctly
   g. If the new problem involves misconceptions the student has shown repeatedly, they are likely to make the same mistake
   
5. Make TWO predictions:
   - predicted_student_option: Based STRICTLY on the student's historical patterns and misconceptions, which option (A, B, C, or D) is the student MOST LIKELY to choose. This should align with their past behavior patterns.
   - predicted_correct_option: Which option (A, B, C, or D) is actually correct based on mathematical reasoning.

IMPORTANT PREDICTION GUIDELINES:
- The predicted_student_option MUST be based on the student's actual historical behavior, not on what would be correct
- If the student has a strong pattern of making a specific type of error, and the new problem presents that same error opportunity, predict that error
- If the student has consistently answered correctly on similar problems, predict they will answer correctly here
- Consider the frequency and recency of misconceptions: more recent and frequent patterns are stronger predictors
- Do NOT simply predict the correct answer for predicted_student_option - predict what the student is actually likely to do based on their history

Output MUST be valid JSON ONLY (no extra text, no markdown, no code blocks). Use this exact schema:
{{
  "history_misconceptions": [
    {{"index": 1, "misconception": "..."}},
    {{"index": 2, "misconception": "..."}},
    ...
    {{"index": 30, "misconception": "..."}}
  ],
  "new_problem_skills": ["skill1", "skill2"],
  "new_option_misconceptions": {{
    "A": "misconception description for option A",
    "B": "misconception description for option B",
    "C": "misconception description for option C",
    "D": "misconception description for option D"
  }},
  "predicted_student_option": "A",
  "predicted_correct_option": "A"
}}"""

# =========================
# Cognitive Load Prompt变体
# =========================
PROMPT_VARIANT_COGNITIVE_LOAD = """Prompt (Cognitive Load Analysis Approach):

You are given a history of 30 math questions, each including:
- Question
- Answer options
- Student's selected answer
- Correct answer

Student's History:

{history_text}

NEW PROBLEM
Question: {new_question}
Options:
A. {new_opt_a}
B. {new_opt_b}
C. {new_opt_c}
D. {new_opt_d}

Instructions:
1. COGNITIVE LOAD ANALYSIS: For each historical question, identify the misconception and assess the cognitive complexity:
   - Simple problems (low cognitive load): basic arithmetic, straightforward concepts
   - Moderate problems (medium cognitive load): multi-step procedures, concept application
   - Complex problems (high cognitive load): abstract reasoning, multiple concepts integration
   - Note: When cognitive load is high, students are more likely to make errors or choose "Not enough information"

2. For the NEW PROBLEM, assess its cognitive load level and identify required skills.

3. For the NEW PROBLEM, for EACH option A/B/C/D, describe what misconception might lead to that choice.

4. COGNITIVE LOAD PREDICTION:
   - If the new problem has LOW cognitive load and the student has mastered the skill, predict CORRECT answer
   - If the new problem has HIGH cognitive load, check if the student has a pattern of struggling with complex problems
   - If the student often chooses "Not enough information" on high-load problems, and this problem is complex, predict that option
   - Match the cognitive load pattern: students tend to perform consistently at similar complexity levels

5. PREDICTIONS:
   - predicted_student_option: Based on cognitive load analysis and student's performance patterns at different complexity levels, what will they choose?
   - predicted_correct_option: Which option is mathematically correct?

Output MUST be valid JSON ONLY:
{{
  "history_misconceptions": [
    {{"index": 1, "misconception": "..."}},
    ...
    {{"index": 30, "misconception": "..."}}
  ],
  "new_problem_skills": ["..."],
  "new_option_misconceptions": {{
    "A": "...",
    "B": "...",
    "C": "...",
    "D": "..."
  }},
  "predicted_student_option": "A",
  "predicted_correct_option": "A"
}}"""

# =========================
# Metacognitive Prompt变体
# =========================
PROMPT_VARIANT_METACOGNITIVE = """Prompt (Metacognitive Awareness Approach):

You are given a history of 30 math questions, each including:
- Question
- Answer options
- Student's selected answer
- Correct answer

Student's History:

{history_text}

NEW PROBLEM
Question: {new_question}
Options:
A. {new_opt_a}
B. {new_opt_b}
C. {new_opt_c}
D. {new_opt_d}

Instructions:
1. METACOGNITIVE ANALYSIS: For each historical question, identify the misconception and assess the student's metacognitive awareness:
   - High awareness: Student answers correctly on problems they understand, avoids guessing
   - Medium awareness: Student sometimes answers correctly, sometimes makes systematic errors
   - Low awareness: Student frequently chooses wrong options, may guess or choose "Not enough information" when uncertain
   - Pattern: Students with low metacognitive awareness often choose the same wrong option type repeatedly

2. For the NEW PROBLEM, identify skills and assess if the student would recognize their own understanding level.

3. For the NEW PROBLEM, for EACH option A/B/C/D, describe what misconception might lead to that choice.

4. METACOGNITIVE PREDICTION:
   - If the student has HIGH metacognitive awareness and the problem matches their skill level, predict CORRECT answer
   - If the student has LOW metacognitive awareness, they may not recognize when they don't understand, leading to systematic errors
   - Check if the student has a pattern of overconfidence (choosing wrong answers confidently) or underconfidence (choosing "Not enough information" too often)
   - Match the metacognitive pattern: students with similar awareness levels tend to behave similarly

5. PREDICTIONS:
   - predicted_student_option: Based on metacognitive awareness analysis, what will the student choose?
   - predicted_correct_option: Which option is mathematically correct?

Output MUST be valid JSON ONLY:
{{
  "history_misconceptions": [
    {{"index": 1, "misconception": "..."}},
    ...
    {{"index": 30, "misconception": "..."}}
  ],
  "new_problem_skills": ["..."],
  "new_option_misconceptions": {{
    "A": "...",
    "B": "...",
    "C": "...",
    "D": "..."
  }},
  "predicted_student_option": "A",
  "predicted_correct_option": "A"
}}"""

# Prompt变体列表
PROMPT_VARIANTS = {
    "default": PROMPT_TEMPLATE,
    "cognitive_load": PROMPT_VARIANT_COGNITIVE_LOAD,
    "metacognitive": PROMPT_VARIANT_METACOGNITIVE,
}


def get_prompt_template(variant: str = "default") -> str:
    """获取指定变体的Prompt模板"""
    return PROMPT_VARIANTS.get(variant, PROMPT_TEMPLATE)


def list_available_variants() -> list:
    """列出所有可用的Prompt变体"""
    return list(PROMPT_VARIANTS.keys())

