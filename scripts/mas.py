import json
import io
import sys
import time
import re
import warnings
import traceback
import textwrap
import torch
import pandas as pd
from typing import List, Dict, Tuple, Optional
from transformers import pipeline
from utils.load_data import load_tabfact_dataset
from tqdm import tqdm
import gc

MAX_NEW_TOKEN = 3000
DEBUG_ROUNDS = 5

print(f"CUDA Version: {torch.version.cuda}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    print(f"  Allocated Memory: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
    print(f"  Cached Memory: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

# Load 3 different models
model_coder = "Qwen/Qwen2.5-Coder-14B-Instruct"
model_instruct = "Qwen/Qwen2.5-7B-Instruct"
model_r1 = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

print(f"Loading coder model: {model_coder}", flush=True)
pipe_coder = pipeline("text-generation", model=model_coder, torch_dtype=torch.float16, device_map="auto")

print(f"Loading instruct model: {model_instruct}", flush=True)
pipe_instruct = pipeline("text-generation", model=model_instruct, torch_dtype=torch.float16, device_map="auto")

print(f"Loading r1 model: {model_r1}", flush=True)
pipe_r1 = pipeline("text-generation", model=model_r1, torch_dtype=torch.float16, device_map="auto")

print(f"DEBUG_ROUNDS: {DEBUG_ROUNDS} and MAX_NEW_TOKEN: {MAX_NEW_TOKEN}")

# ----------- Helper functions from original code -------------
def extract_user_traceback(tb: str) -> str:
    lines = tb.strip().split('\n')
    user_lines = [line for line in lines if '<string>' in line or line.strip().startswith((
        'Traceback', 'KeyError', 'ValueError', 'TypeError', 'IndexError', 'NameError', 'AttributeError'))]
    return '\n'.join(user_lines)

def get_error_line_number(tb: str) -> int:
    match = re.search(r'File "<string>", line (\d+)', tb)
    if match:
        return int(match.group(1))
    return -1

# ----------- Updated extraction and normalization from provided code -----------
def extract_code(text: str) -> str:
    text = text.strip()

    # Step 1: Extract code blocks inside triple backticks with optional language
    code_blocks = re.findall(
        r"```(?:\w+)?\n(.*?)```",  # non-greedy code block
        text,
        re.DOTALL | re.IGNORECASE,
    )

    if code_blocks:
        # Use the largest code block by character length
        code = max(code_blocks, key=len)
    else:
        # Fallback: treat whole text as possible code
        code = text

    # Step 2: Dedent and clean up
    code = textwrap.dedent(code).strip()

    # Step 3: Remove trailing hallucinated explanations
    hallucination_markers = [
        r"final answer", r"explanation", r"summary", r"so ", r"therefore", 
        r"we conclude", r"in conclusion", r"the answer is", r"thus", 
        r"as a result", r"conclusion", r"the output is", r"print result", r"answer:"
    ]
    pattern = r'(?i)(?:' + '|'.join(hallucination_markers) + r')\b'
    code = re.split(pattern, code)[0].strip()

    # Step 4: Clip after last print() if it exists
    print_matches = list(re.finditer(r'print\s*$([^()]*|\([^()]*$)*\)', code))
    if print_matches:
        last_print_end = print_matches[-1].end()
        code = code[:last_print_end]

    # Step 5: Final cleanup of lines
    lines = code.splitlines()
    lines = [line.rstrip() for line in lines if line.strip()]
    code = '\n'.join(lines).strip()

    # Step 6: Robust sanity checks
    def looks_complete(code: str) -> bool:
        # Heuristics: balanced parens, colons on defs/ifs, etc.
        if code.count('(') != code.count(')'):
            return False
        if code.count('[') != code.count(']'):
            return False
        if code.count('{') != code.count('}'):
            return False
        if re.search(r'\bdef\b.*[^:]\n', code):  # def line missing ':'
            return False
        return True

    valid_starts = ('import', 'from', 'def', 'class', '#', 'data =', 'df =', 'table =', 'try:', 'pd.')
    if not code.startswith(valid_starts) or not looks_complete(code):
        print("⚠️ Warning: Extracted code may be invalid, incomplete, or poorly formatted", flush=True)

    return code

def execute_code(code: str) -> Tuple[str, str]:
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            compile(code, "<string>", "exec")
            exec(code, {
                "pd": pd, "json": json,
                "exit": lambda *a, **k: (_ for _ in ()).throw(RuntimeError("exit() called"))
            })
        output = new_stdout.getvalue().strip()
        return output if output else "No output", None
    except Exception as e:
        return "", f"{e.__class__.__name__}: {str(e)}"
    finally:
        sys.stdout = old_stdout

def normalize_prediction(pred: str) -> int:
    text = pred.lower()

    # Layer 1: strict pattern match
    match = re.search(r'final answer\s*:?\s*(true|false|unknown)', text)
    if match:
        return {"true": 1, "false": 0, "unknown": 2}[match.group(1)]

    # Layer 2: fuzzy match on trailing words
    last_part = " ".join(text.split()[-30:])
    last_part = re.sub(r'[^\w\s]', '', last_part)

    if "false" in last_part: return 0
    if "true" in last_part: return 1
    if "unknown" in last_part: return 2

    # Layer 3: rule-based weak signals
    if re.search(r'(correct answer|conclusion|determination)\s*(is|:)?\s*false', text):
        return 0
    if re.search(r'(correct answer|conclusion|determination)\s*(is|:)?\s*true', text):
        return 1

    # Layer 4: frequency voting
    score = {"true": text.count("true"), "false": text.count("false"), "unknown": text.count("unknown")}
    if score["false"] > max(score["true"], score["unknown"]):
        return 0
    if score["true"] > max(score["false"], score["unknown"]):
        return 1
    if score["unknown"] > 0:
        return 2

    # Completely uncertain
    return 2

def extract_code_prediction(code_output: str) -> int:
    """Extract prediction from code output by looking for True/False patterns"""
    if not code_output or code_output.strip() == "No output":
        return 2  # Unknown
    
    text = code_output.lower()
    
    # Look for explicit True/False in output
    if "true" in text and "false" not in text:
        return 1
    elif "false" in text and "true" not in text:
        return 0
    
    # Look for numerical patterns that might indicate True/False
    lines = code_output.strip().split('\n')
    for line in lines:
        line = line.strip().lower()
        if 'true' in line:
            return 1
        elif 'false' in line:
            return 0
    
    return 2  # Unknown if can't determine

def query_llm(pipe, prompt: str, max_tokens: int = MAX_NEW_TOKEN) -> str:
    try:
        response = pipe(prompt, max_new_tokens=max_tokens)[0]["generated_text"]
        result = response[len(prompt):].strip()
        return result
    except Exception as e:
        return f"# Generation error: {e}"

def append_jsonl(result: Dict, file_path: str):
    with open(file_path, 'a', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)
        f.write('\n')

def log_to_file(content: str, log_file: str):
    """Append content to log file"""
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(content + '\n')

# ----------- Analysis Functions -------------
def analyze_agreement(results: List[Dict]) -> Dict:
    """Analyze agreement between reasoning and code predictions"""
    stats = {
        "total_samples": len(results),
        "reasoning_correct": 0,
        "code_correct": 0,
        "verifier_correct": 0,
        "both_correct": 0,
        "both_incorrect": 0,
        "agree_and_correct": 0,
        "agree_but_incorrect": 0,
        "disagree_reasoning_correct": 0,
        "disagree_code_correct": 0,
        "final_correct": 0,
        "agreement_rate": 0.0,
        "reasoning_accuracy": 0.0,
        "code_accuracy": 0.0,
        "verifier_accuracy": 0.0,
        "final_accuracy": 0.0
    }
    
    for result in results:
        label = int(result["label"])
        reasoning_pred = result["reasoning_prediction"]
        code_pred = result["code_prediction"] 
        verifier_pred = result["verifier_prediction"]
        final_pred = result["prediction"]
        
        # Individual correctness
        reasoning_correct = (reasoning_pred == label)
        code_correct = (code_pred == label)
        verifier_correct = (verifier_pred == label)
        final_correct = (final_pred == label)
        
        if reasoning_correct:
            stats["reasoning_correct"] += 1
        if code_correct:
            stats["code_correct"] += 1
        if verifier_correct:
            stats["verifier_correct"] += 1
        if final_correct:
            stats["final_correct"] += 1
            
        # Agreement analysis
        agree = (reasoning_pred == code_pred)
        
        if reasoning_correct and code_correct:
            stats["both_correct"] += 1
        elif not reasoning_correct and not code_correct:
            stats["both_incorrect"] += 1
            
        if agree:
            if reasoning_correct and code_correct:
                stats["agree_and_correct"] += 1
            elif not reasoning_correct and not code_correct:
                stats["agree_but_incorrect"] += 1
        else:  # disagree
            if reasoning_correct:
                stats["disagree_reasoning_correct"] += 1
            elif code_correct:
                stats["disagree_code_correct"] += 1
    
    # Calculate rates
    if stats["total_samples"] > 0:
        stats["reasoning_accuracy"] = stats["reasoning_correct"] / stats["total_samples"]
        stats["code_accuracy"] = stats["code_correct"] / stats["total_samples"] 
        stats["verifier_accuracy"] = stats["verifier_correct"] / stats["total_samples"]
        stats["final_accuracy"] = stats["final_correct"] / stats["total_samples"]
        
        # Agreement rate (both methods produce same prediction)
        agreement_count = sum(1 for r in results if r["reasoning_prediction"] == r["code_prediction"])
        stats["agreement_rate"] = agreement_count / stats["total_samples"]
    
    return stats

def print_analysis(stats: Dict, log_file: str = None):
    """Print detailed analysis of results"""
    analysis_text = "\n" + "="*50 + "\n"
    analysis_text += "DETAILED ANALYSIS\n"
    analysis_text += "="*50 + "\n"
    analysis_text += f"Total Samples: {stats['total_samples']}\n"
    analysis_text += f"Final Accuracy: {stats['final_accuracy']:.3f}\n"
    analysis_text += f"Reasoning Accuracy: {stats['reasoning_accuracy']:.3f}\n"
    analysis_text += f"Code Accuracy: {stats['code_accuracy']:.3f}\n"
    analysis_text += f"Verifier Accuracy: {stats['verifier_accuracy']:.3f}\n"
    analysis_text += f"Agreement Rate: {stats['agreement_rate']:.3f}\n"
    analysis_text += "\n"
    analysis_text += "AGREEMENT BREAKDOWN:\n"
    analysis_text += f"  Both methods correct: {stats['both_correct']}\n"
    analysis_text += f"  Both methods incorrect: {stats['both_incorrect']}\n"
    analysis_text += f"  Agree and correct: {stats['agree_and_correct']}\n"
    analysis_text += f"  Agree but incorrect: {stats['agree_but_incorrect']}\n"
    analysis_text += f"  Disagree - reasoning correct: {stats['disagree_reasoning_correct']}\n"
    analysis_text += f"  Disagree - code correct: {stats['disagree_code_correct']}\n"
    analysis_text += "="*50
    
    print(analysis_text)
    if log_file:
        log_to_file(analysis_text, log_file)

# ----------- Agent Classes -------------

class PlannerAgent:
    def __init__(self, pipe):
        self.pipe = pipe
    
    def create_plan(self, table: str, statement: str) -> str:
        prompt = f"""You are a Planning Agent.

You are given:
- A table in JSON format.
- A natural language statement.

=== Table ===
{table}

=== Statement ===
{statement}

Your task is to write step-by-step logical plan to verify whether the statement is true or false, using only natural language. However, you must not output the final answer or conclusion.

Carefully identify what is being claimed. Pay attention to key quantities, conditions, comparisons, or relationships involved. Explain the table structure and meaning of rows/columns as needed. 

Your plan should include:
- Any necessary filters or conditions applied to rows (e.g., "Filter rows where ...").
- Any computations needed (e.g., "Count number of rows where ...", "Compute sum of ...", "Compute average of ...").
- Any comparisons or thresholds being checked (e.g., "Compare count to threshold ...").
- Any derived intermediate values you compute.

If you need to reference information from the table, describe which columns, values, or conditions are involved in natural language. Be precise about what information is needed from the table.

### Important Constraints:
- DO NOT mention pandas, DataFrame, or any code-related terms.
- DO NOT write or suggest code.
- Stay fully in natural language reasoning.
"""
        response = query_llm(self.pipe, prompt, max_tokens=512)
        return response

class ReasoningAgent:
    def __init__(self, pipe):
        self.pipe = pipe
    
    def analyze(self, table: str, statement: str) -> Tuple[str, int]:
        prompt = f"""You are a Reasoning Agent for Fact Verification.

You are given a table and a statement. Your task is to determine whether the following statement is true or false based on the table. Justify your decision and output a confidence score (between 0 and 1) representing your certainty.

=== Table ===
{table}

=== Statement ===
{statement}

Respond in the following format:
Justification: <Your short reasoning>
Confidence: <float between 0 and 1>
Final Answer: True / False / Unknown
"""
        response = query_llm(self.pipe, prompt, max_tokens=5000)
        prediction = normalize_prediction(response)
        return response, prediction

class CoderAgent:
    def __init__(self, pipe, enable_auto_debug=True):
        self.pipe = pipe
        self.enable_auto_debug = enable_auto_debug
    
    def build_code_prompt(self, table: str, statement: str, reasoning_guidance: str = None, 
                         previous_code: str = None, error_msg: str = None) -> str:
        base = f"""You are a Python pandas code writer.

You are given:
- Table data in JSON format
- A natural language statement

Your task is to write Python code that:
- Loads the table using pandas
- Implements logic to verify the statement
- Determine whether the statement is true or false.

=== Table ===
{table}

=== Statement ===
{statement}
"""
        
        # Add reasoning guidance if provided
        if reasoning_guidance:
            base += f"\n=== Reasoning Guidance ===\n{reasoning_guidance}\n"
        
        # Different instructions based on whether planner guidance is available
        if reasoning_guidance:
            # With planner - follow the guidance
            base += """
Instructions:
1. Follow the reasoning guidance step-by-step and output executable Python code:
   - The code must start with 'import pandas as pd'.
   - Use Python comments (#) for any explanation.
   - Do not output plain answers or conclusions outside code.
   - Implement each step mentioned in the reasoning guidance systematically.

2. Load the table as a pandas DataFrame:
   - The first row is the header; the rest are data rows.
   - Use `pd.DataFrame(..., columns=...)` or similar logic to build the DataFrame cleanly.
   - Ensure proper alignment of column headers and data.

3. Implement the reasoning plan:
   - Follow the logical steps outlined in the reasoning guidance.
   - For each step mentioned in the guidance, implement the corresponding pandas operations.
   - Use the guidance to determine what filters, computations, and comparisons are needed.

4. Safely preprocess and transform the data:
   - Use `pd.to_numeric(..., errors='coerce')` to convert columns for numeric operations.
   - Before applying any operation (e.g., arithmetic, string, comparison), check that the operand types are compatible.
   - Use `.copy()` when modifying filtered DataFrames to avoid warnings.
   - Always check if the DataFrame is empty before accessing rows by index (`df.empty` or `len(df) == 0`).
   - Handle missing values and malformed entries by filling, dropping, or filtering as appropriate.

5. As you implement each step from the reasoning guidance, print intermediate results:
   - For every key step mentioned in the guidance, print helpful evidence.
   - Use clear, machine-readable prints such as:
       - print("filter_rows:", number_of_rows_remaining)
       - print("sum_column_X:", computed_sum)
       - print("condition_X_met:", boolean_result)
   - These prints should correspond to the steps outlined in the reasoning guidance.

6. IMPORTANT: End your code with a clear final answer:
   - Print the final result as: print("FINAL_ANSWER:", True) or print("FINAL_ANSWER:", False)
   - This should be the very last line of your code.
"""
        else:
            # Without planner - analyze and implement independently
            base += """
Instructions:
1. Output only executable Python code:
   - The output must start with `import`.
   - Do NOT include markdown (e.g., ```python).
   - Any explanation must be in Python comments (`#`).
   - Do not return plain answers like "True" or "False"—only `print("true")` or `print("false")` is allowed.

2. Load the table as a pandas DataFrame:
   - The first row is the header; the rest are data rows.
   - Use `pd.DataFrame(..., columns=...)` or similar logic to build the DataFrame cleanly.
   - Ensure proper alignment of column headers and data.

3. Safely preprocess and transform the data:
   - Use `pd.to_numeric(..., errors='coerce')` to convert columns for numeric operations.
   - Before applying any operation (e.g., arithmetic, string, comparison), check that the operand types are compatible.
     - Example: Use `isinstance(x, (int, float))` or `if pd.api.types.is_numeric_dtype(...)` to avoid `TypeError`.
   - Use `.copy()` when modifying filtered DataFrames to avoid warnings.
   - Always check if the DataFrame is empty before accessing rows by index (`df.empty` or `len(df) == 0`).
   - When inserting new columns, ensure the index aligns with the existing DataFrame (`df.reset_index(drop=True)` if needed).
   - Handle missing values and malformed entries by filling, dropping, or filtering as appropriate.

4. Implement robust logic:
   - Reason step-by-step and reflect each logical step in the code.
   - Do NOT hardcode the final result—derive it from the DataFrame using calculations and conditions.
   - Use control flow and safe condition checks to guide the logic.

5. Handle edge cases and avoid runtime errors:
   - Guard against `KeyError`, `IndexError`, `AttributeError`, `ValueError`, and `TypeError`.
   - Use `if 'column' in df.columns` before accessing a column.
   - Use `if len(df) > 0` or `if not df.empty` before indexing.
   - Use `try`/`except` blocks where necessary, especially around risky operations.

6. Output:
   - Print important intermediate calculation results for debugging and verification.
   - Your final output must be either `print("true")` or `print("false")`—lowercase only.
   - If the table is empty or missing key data, output `print("No data")` and exit gracefully.

7. Your entire output must be a single, self-contained, error-free Python script.
"""
        
        # Add debugging instructions if this is a retry
        if previous_code and error_msg:
            base += f"""

# DEBUGGING MODE: The previous code attempt failed with an error.
# Previous Code:
{textwrap.indent(previous_code, "# ")}

# Error:
# {error_msg}

# Instructions for debugging:
# 1. Carefully analyze the error message above.
# 2. Identify the root cause of the failure (syntax, logic, data type issues, etc.).
# 3. Before writing new code, add comments explaining what went wrong and how you'll fix it.
# 4. Revise the code to address the specific error while maintaining the original logic.
# 5. Ensure that:
#    - Operand types are compatible (e.g., avoid adding string to int).
#    - DataFrame index and shape are preserved when transforming or assigning columns.
#    - All column references and row indices are valid.
#    - The result is logically derived and printed as either "true" or "false".
# 6. Be more defensive in your coding - add type checks and error handling where appropriate.
"""
        
        return base
    
    def generate_code(self, table: str, statement: str, reasoning_guidance: str = None) -> Tuple[str, str, str, Optional[str], List[Dict], int]:
        prompt = self.build_code_prompt(table, statement, reasoning_guidance)
        raw_code = query_llm(self.pipe, prompt, max_tokens=5000)
        code = extract_code(raw_code)
        output, error = execute_code(code)
        
        debug_attempts = [{"code": code, "output": output, "error": error}]
        
        # Auto-debug if enabled and there's an error or invalid output
        if self.enable_auto_debug:
            for debug_round in range(DEBUG_ROUNDS):
                invalid_output = (not output.strip()) or (output.strip().lower() == "no data")
                code_invalid = code.startswith("# invalid code") or "⚠️ Warning:" in code
                
                if not error and not invalid_output and not code_invalid:
                    break  # Only stop if code runs, produces meaningful output, and starts validly
                
                print(f"Debug Round {debug_round + 1}: Fixing code issues...")
                prompt = self.build_code_prompt(table, statement, reasoning_guidance, code, 
                                              error or "Output was empty or invalid")
                raw_code = query_llm(self.pipe, prompt, max_tokens=5000)
                code = extract_code(raw_code)
                output, error = execute_code(code)
                debug_attempts.append({"code": code, "output": output, "error": error})
        
        # Extract prediction from code output
        if error:
            code_prediction = 2  # Unknown due to error
        else:
            code_prediction = extract_code_prediction(output)
        
        return raw_code, code, output, error, debug_attempts, code_prediction

class VerifierAgent:
    def __init__(self, pipe):
        self.pipe = pipe
    
    def verify(self, table: str, statement: str, reasoning_output: str, reasoning_prediction: int,
              code: str, output: str, error: Optional[str], code_prediction: int) -> Tuple[str, int]:
        prompt = f"""You are a Verification Agent combining symbolic reasoning and execution analysis.

You are given:
- A table
- A statement
- Reasoner's analysis and prediction
- Code implementing logic over the table
- Output from the code (print statements or errors)

Your task is to determine whether the following statement is true or false using the following inputs:

=== Table ===
{table}

=== Statement ===
{statement}

=== Reasoning Analysis ===
{reasoning_output}
Reasoning Prediction: {reasoning_prediction} (0=False, 1=True, 2=Unknown)

=== Code ===
{code}

=== Code Output ===
{output if not error else f"Execution Error: {error}"}
Code Prediction: {code_prediction} (0=False, 1=True, 2=Unknown)

Use the following steps:
1. Interpret the reasoner's conclusion and confidence.
2. Analyze the code and its output for correctness and relevance.
3. Cross-check the table, if needed, to resolve ambiguity or contradiction.
4. If reasoner and code agree, follow their joint verdict.
5. If they disagree, weigh the trustworthiness (confidence, correctness) of each and make a final judgment.

Finally, conclude with:
Justification: <Your reasoning for the final decision>
Final Answer: True / False / Unknown
"""
        
        response = query_llm(self.pipe, prompt, max_tokens=1024)
        prediction = normalize_prediction(response)
        
        return response, prediction

# ----------- Main Multi-Agent System -------------

class MultiAgentSystem:
    def __init__(self, pipe_coder, pipe_instruct, pipe_r1, enable_auto_debug=True, use_planner=True):
        self.use_planner = use_planner
        if use_planner:
            self.planner_agent = PlannerAgent(pipe_instruct)
        self.reasoning_agent = ReasoningAgent(pipe_r1)
        self.coder_agent = CoderAgent(pipe_coder, enable_auto_debug)
        self.verifier_agent = VerifierAgent(pipe_instruct)
    
    def process_sample(self, table: str, statement: str, label: str, log_file: str) -> Dict:
        # Log sample info
        log_content = f"\n{'='*80}\nSample Processing\n{'='*80}\n"
        log_content += f"Statement: {statement}\n"
        log_content += f"Label: {label}\n"
        log_content += f"Table: {table}\n\n"
        
        reasoning_guidance = None
        
        # Step 1: Optional Planner Agent
        if self.use_planner:
            print("Step 1: Planner Agent creating reasoning plan...")
            log_content += "STEP 1: PLANNER AGENT\n" + "-"*40 + "\n"
            reasoning_guidance = self.planner_agent.create_plan(table, statement)
            log_content += f"Reasoning Guidance:\n{reasoning_guidance}\n\n"
        
        # Step 2: Reasoning Agent analyzes the problem
        step_num = 2 if self.use_planner else 1
        print(f"Step {step_num}: Reasoning Agent analyzing...")
        log_content += f"STEP {step_num}: REASONING AGENT\n" + "-"*40 + "\n"
        reasoning_output, reasoning_prediction = self.reasoning_agent.analyze(table, statement)
        log_content += f"Reasoning Output:\n{reasoning_output}\n"
        log_content += f"Reasoning Prediction: {reasoning_prediction}\n\n"
        
        # Step 3: Coder Agent generates and potentially debugs code
        step_num = 3 if self.use_planner else 2
        print(f"Step {step_num}: Coder Agent generating code...")
        log_content += f"STEP {step_num}: CODER AGENT\n" + "-"*40 + "\n"
        raw_code, code, code_output, code_error, debug_attempts, code_prediction = self.coder_agent.generate_code(
            table, statement, reasoning_guidance
        )
        log_content += f"Raw Code:\n{raw_code}\n\n"
        log_content += f"Extracted Code:\n{code}\n\n"
        log_content += f"Code Output:\n{code_output if not code_error else f'ERROR: {code_error}'}\n"
        log_content += f"Code Prediction: {code_prediction}\n"
        if len(debug_attempts) > 1:
            log_content += f"Debug Attempts: {len(debug_attempts)}\n"
            for i, attempt in enumerate(debug_attempts):
                status = "Success" if not attempt["error"] else f"Error: {attempt['error']}"
                log_content += f"  Attempt {i+1}: {status}\n"
        log_content += "\n"
        
        # Step 4: Verifier Agent makes final conclusion
        step_num = 4 if self.use_planner else 3
        print(f"Step {step_num}: Verifier Agent making final conclusion...")
        log_content += f"STEP {step_num}: VERIFIER AGENT\n" + "-"*40 + "\n"
        verifier_output, verifier_prediction = self.verifier_agent.verify(
            table, statement, reasoning_output, reasoning_prediction,
            code, code_output, code_error, code_prediction
        )
        log_content += f"Verifier Output:\n{verifier_output}\n"
        log_content += f"Verifier Prediction: {verifier_prediction}\n\n"
        
        # Determine correctness for each agent
        label_int = int(label)
        reasoning_correct = (reasoning_prediction == label_int)
        code_correct = (code_prediction == label_int)
        verifier_correct = (verifier_prediction == label_int)
        methods_agree = (reasoning_prediction == code_prediction)
        
        # Use verifier prediction as final prediction
        final_prediction = verifier_prediction
        final_correct = verifier_correct
        
        # Log results
        log_content += "RESULTS\n" + "-"*40 + "\n"
        log_content += f"Reasoning Correct: {reasoning_correct}\n"
        log_content += f"Code Correct: {code_correct}\n"
        log_content += f"Verifier Correct: {verifier_correct}\n"
        log_content += f"Methods Agree: {methods_agree}\n"
        log_content += f"Final Prediction: {final_prediction}\n"
        log_content += f"Final Correct: {final_correct}\n"
        
        log_to_file(log_content, log_file)
        
        # Return compact result for JSON
        result = {
            "sample_id": 0,  # Will be set in main loop
            "statement": statement,
            "label": int(label),
            "reasoning_prediction": reasoning_prediction,
            "reasoning_correct": reasoning_correct,
            "code_prediction": code_prediction,
            "code_correct": code_correct,
            "verifier_prediction": verifier_prediction,
            "verifier_correct": verifier_correct,
            "methods_agree": methods_agree,
            "prediction": final_prediction,
            "match": final_correct,
            "has_planner": self.use_planner,
            "debug_attempts": len(debug_attempts),
            "has_error": code_error is not None
        }
        
        return result

# ----------- Main Execution -------------

if __name__ == "__main__":
    dataset_path = "data/tabfact/test.jsonl"
    raw2clean_path = "data/tabfact/raw2clean.jsonl"
    output_file = "mas_a_nop.jsonl"
    log_file = "mas_a_nop.txt"
    first_n = 20
    start_idx = 0
    # Option for autodebug and planner agent
    enable_auto_debug = True  # Set to False to disable auto-debugging
    use_planner = False  # Set to False to disable planner agent

    print(f"Initializing Multi-Agent System (Auto-debug: {enable_auto_debug}, Planner: {use_planner})...")
    
    # Initialize log file
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Multi-Agent System Log\n")
        f.write(f"Auto-debug: {enable_auto_debug}\n")
        f.write(f"Use Planner: {use_planner}\n")
        f.write(f"Models: Coder={model_coder}, Instruct={model_instruct}, R1={model_r1}\n")
        f.write(f"{'='*80}\n")
    
    multi_agent_system = MultiAgentSystem(pipe_coder, pipe_instruct, pipe_r1, enable_auto_debug, use_planner)

    dataset = load_tabfact_dataset(dataset_path, raw2clean_path, first_n=first_n)
    final_results = []

    for i, sample in enumerate(tqdm(dataset[start_idx:], desc="Processing with Multi-Agent System")):
        table, statement, label = sample["table_text"], sample["statement"], sample["label"]

        print(f"\n===== Sample {i + start_idx + 1} =====")
        print(f"Statement: {statement}")
        
        result = multi_agent_system.process_sample(table, statement, label, log_file)
        result["sample_id"] = i + start_idx
        
        print(f"Reasoning Prediction: {result['reasoning_prediction']} (Correct: {result['reasoning_correct']})")
        print(f"Code Prediction: {result['code_prediction']} (Correct: {result['code_correct']})")
        print(f"Verifier Prediction: {result['verifier_prediction']} (Correct: {result['verifier_correct']})")
        print(f"Methods Agree: {result['methods_agree']}")
        print(f"Final: {result['prediction']}, Label: {label}, Match: {result['match']}")
        print("-" * 100)

        final_results.append(result)
        append_jsonl(result, output_file)

        # Periodic analysis
        if (i + 1) % 5 == 0:
            stats = analyze_agreement(final_results)
            intermediate_log = f"\n[{i + 1}] INTERMEDIATE RESULTS:\n"
            intermediate_log += f"  Final Accuracy: {stats['final_accuracy']:.3f}\n"
            intermediate_log += f"  Reasoning Accuracy: {stats['reasoning_accuracy']:.3f}\n"
            intermediate_log += f"  Code Accuracy: {stats['code_accuracy']:.3f}\n"
            intermediate_log += f"  Verifier Accuracy: {stats['verifier_accuracy']:.3f}\n"
            intermediate_log += f"  Agreement Rate: {stats['agreement_rate']:.3f}\n"
            
            print(intermediate_log)
            log_to_file(intermediate_log, log_file)
            clear_gpu_memory()

    # Final comprehensive analysis
    final_stats = analyze_agreement(final_results)
    print_analysis(final_stats, log_file)
    
    # Log final summary
    final_summary = f"\nFINAL SUMMARY\n{'='*50}\n"
    final_summary += f"Total Samples Processed: {len(final_results)}\n"
    final_summary += f"Configuration: Auto-debug={enable_auto_debug}, Planner={use_planner}\n"
    final_summary += f"Models Used:\n"
    final_summary += f"  - Coder: {model_coder}\n"
    final_summary += f"  - Instruct: {model_instruct}\n"
    final_summary += f"  - R1: {model_r1}\n"
    final_summary += f"Results saved to: {output_file}\n"
    final_summary += f"Detailed log saved to: {log_file}\n"
    
    print(final_summary)
    log_to_file(final_summary, log_file)
    
    print(f"\nFinished processing {len(final_results)} samples.")
    print(f"Compact results saved to {output_file}")
    print(f"Detailed log saved to {log_file}")
    
    # Save summary statistics
    summary_file = output_file.replace('.jsonl', '_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(final_stats, f, indent=2)
    print(f"Summary statistics saved to {summary_file}")