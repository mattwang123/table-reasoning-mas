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
from typing import List, Dict, Tuple
from transformers import pipeline
from utils.load_data import load_tabfact_dataset
from tqdm import tqdm

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

model_id = "Qwen/Qwen2.5-Coder-14B-Instruct"

print(f"DEBUG_ROUNDS: {DEBUG_ROUNDS} and MAX_NEW_TOKEN: {MAX_NEW_TOKEN}")
print(f"Loading model {model_id} ...")
start_time = time.time()
try:
    pipe = pipeline(
        "text-generation", 
        model=model_id, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print(f"Model loaded successfully in {time.time() - start_time:.2f} seconds.")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# ----------- New helper functions -------------
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
# ----------------------------------------------

def extract_python_code(text: str) -> str:
    text = text.strip()
    text = re.sub(r'^```(?:python)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text)
    text = re.split(r'(?i)\n\s*(final answer|explanation|output|the answer is|so|therefore)\b', text)[0]
    code = textwrap.dedent(text).strip()

    print_matches = list(re.finditer(r'print\s*\(([^()]*|\([^()]*\))*\)', code))
    if print_matches:
        last_print_end = print_matches[-1].end()
        code = code[:last_print_end]

    lines = code.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and not lines[-1].strip().endswith((')', '}', '"', "'")):
        lines[-1] += ')'

    code = '\n'.join(lines)
    valid_starts = ('import', 'from', 'df =', '#', 'data =', 'table =', 'try:')
    if not code.startswith(valid_starts):
        return "# invalid code: does not start with valid prefix"
    return code

def execute_code(code: str) -> Tuple[str, str]:
    clean_code = extract_python_code(code)
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            compiled = compile(clean_code, "<string>", "exec")
            exec(compiled, {
                "pd": pd, "json": json,
                "exit": lambda *a, **k: (_ for _ in ()).throw(RuntimeError("exit() called"))
            })
        output = new_stdout.getvalue().strip()
        return output if output else "No output", None
    except Exception:
        tb = traceback.format_exc()
        user_tb = extract_user_traceback(tb)
        line_num = get_error_line_number(tb)
        detailed_error = f"[Line {line_num}] {user_tb.strip()}" if line_num > 0 else user_tb.strip()
        return "", detailed_error
    finally:
        sys.stdout = old_stdout


def normalize_prediction(pred: str) -> int:
    p = pred.strip().lower()
    if p in ["1", "true"]:
        return 1
    elif p in ["0", "false"]:
        return 0
    return 2

def build_prompt(table: str, statement: str, previous_code: str = None, error_msg: str = None) -> str:
    base = f"""
You are given a table in JSON format followed by a natural language statement. Your task is to write Python code that analyzes the table to determine whether the statement is true or false.

Table: {table}
Statement: {statement}

Follow these instructions carefully to generate robust and correct code:

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
   - Your final output must be either `print("true")` or `print("false")`—lowercase only.
   - If the table is empty or missing key data, output `print("No data")` and exit gracefully.

7. Your entire output must be a single, self-contained, error-free Python script.
"""
    if previous_code and error_msg:
        base += f"""
# The previous code attempt failed with an error. Please analyze the issue, revise the code accordingly, and ensure it runs correctly.
# Code:
{textwrap.indent(previous_code, "# ")}

# Error:
# {error_msg}

# Instructions:
# Carefully analyze the error. Before writing new code, reflect on the possible cause(s) of failure.
# Then revise the code to fix the issue, ensuring that:
# - Operand types are compatible (e.g., avoid adding string to int).
# - DataFrame index and shape are preserved when transforming or assigning columns.
# - All column references and row indices are valid.
# - The result is logically derived and printed as either "true" or "false".
# Be deliberate and cautious. Explain your reasoning via comments if needed.
"""
    return base

def query_llm(pipe, prompt: str) -> str:
    try:
        response = pipe(prompt, max_new_tokens=MAX_NEW_TOKEN)[0]["generated_text"]
        code = response[len(prompt):].strip()
        return extract_python_code(code)
    except Exception as e:
        return f"# Generation error: {e}"

def append_jsonl(result: Dict, file_path: str):
    with open(file_path, 'a', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)
        f.write('\n')

if __name__ == "__main__":
    dataset_path = "data/tabfact/test.jsonl"
    raw2clean_path = "data/tabfact/raw2clean.jsonl"
    output_file = "autodebug_qwen14_trace_valid.jsonl"
    first_n = -1
    start_idx = 0

    print(f"Loading model {model_id} ...", flush=True)
    pipe = pipeline(
        "text-generation", 
        model=model_id, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )

    dataset = load_tabfact_dataset(dataset_path, raw2clean_path, first_n=first_n)
    final_results = []

    for i, sample in enumerate(tqdm(dataset[start_idx:], desc="Auto Debugging Samples")):
        table, statement, label = sample["table_text"], sample["statement"], sample["label"]

        prompt = build_prompt(table, statement)
        code = query_llm(pipe, prompt)
        output, error = execute_code(code)

        debug_attempts = [{"code": code, "output": output, "error": error}]

        print(f"\n=== Sample {i + 1} ===", flush=True)
        print(f"Statement: {statement}", flush=True)
        print(f"Initial Code:\n{textwrap.indent(code, '    ')}", flush=True)
        if error:
            print(f"Initial Error: {error}", flush=True)
        else:
            print("Initial execution succeeded.", flush=True)

        for debug_round in range(DEBUG_ROUNDS):
            # Treat "no output" or "no data" as failed execution
            invalid_output = (not output.strip()) or (output.strip().lower() == "no data")
            code_invalid = code.startswith("# invalid code")
            if not error and not invalid_output and not code_invalid:
                break  # Only stop if code runs, produces meaningful output, and starts validly

            print(f"\n--- Debug Round {debug_round + 1} ---", flush=True)
            print("Previous Code:", flush=True)
            print(textwrap.indent(code, "    "), flush=True)
            if error:
                print(f"Error: {error}", flush=True)
            elif invalid_output:
                print("Invalid output (empty or 'No data') — will continue debugging.", flush=True)
            elif code_invalid:
                print("Code format invalid — will continue debugging.", flush=True)
            else:
                print("Unexpected debug condition — continuing.", flush=True)

            prompt = build_prompt(table, statement, code, error or "Output was empty or invalid")
            code = query_llm(pipe, prompt)
            output, error = execute_code(code)
            debug_attempts.append({"code": code, "output": output, "error": error})

        prediction = normalize_prediction(output)
        match = prediction == int(label)

        print(f"\nFinal Code:\n{textwrap.indent(code, '    ')}", flush=True)
        print(f"Final Output: {output}", flush=True)
        if error:
            print(f"Final Error: {error}", flush=True)
        print(f"Prediction: {prediction}, Label: {label}, Match: {match}", flush=True)
        print("-" * 100, flush=True)

        result = {
            "statement": statement,
            "label": label,
            "final_code": code,
            "prediction": prediction,
            "match": match,
            "output": output,
            "error": error,
            "attempts": debug_attempts
        }

        final_results.append(result)
        append_jsonl(result, output_file)

        if (i + 1) % 10 == 0:
            acc = 100 * sum(r["match"] for r in final_results) / len(final_results)
            print(f"[{i + 1}] Intermediate Accuracy: {acc:.2f}%", flush=True)

    final_accuracy = 100 * sum(r["match"] for r in final_results) / len(final_results)
    print(f"\nFinished processing {len(final_results)} samples.", flush=True)
    print(f"Final Accuracy: {final_accuracy:.2f}%", flush=True)
    print(f"Saved all results to {output_file}", flush=True)
