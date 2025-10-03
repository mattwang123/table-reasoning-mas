import torch
import json
import re
import io
import sys
import warnings
import traceback
import textwrap
import pandas as pd
import subprocess
import tempfile
import os
from typing import Dict, Any, Optional, Tuple, List
from vllm import LLM, SamplingParams

def load_single_model_vllm(model_path: str, model_name: str) -> LLM:
    """Load a single VLLM model with conservative settings"""
    
    print(f"Loading {model_name}: {model_path}")
    
    # Clear GPU memory before loading
    torch.cuda.empty_cache()
    
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.7,
        trust_remote_code=True,
        max_model_len=4096,
        disable_log_stats=True,
        max_num_seqs=16,
    )
    
    # Print memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        cached = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"  {model_name} loaded: {allocated:.2f} GB allocated, {cached:.2f} GB cached")
    
    return llm

def unload_model(llm: LLM):
    """Unload model and clear GPU memory"""
    del llm
    torch.cuda.empty_cache()
    print("  Model unloaded and GPU memory cleared")

def query_llm_vllm(llm: LLM, prompt: str, max_tokens: int = 1000) -> str:
    """Query VLLM model"""
    
    sampling_params = SamplingParams(
        temperature=0.3,  # Changed from 0.1 to 0.3
        top_p=0.9,
        max_tokens=max_tokens,
        stop=None,
    )
    
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text

# ----------- EXACT SAME HELPER FUNCTIONS FROM ORIGINAL -----------

def extract_user_traceback(tb: str) -> str:
    """Extract user-relevant traceback lines - EXACT SAME as original"""
    lines = tb.strip().split('\n')
    user_lines = [line for line in lines if '<string>' in line or line.strip().startswith((
        'Traceback', 'KeyError', 'ValueError', 'TypeError', 'IndexError', 'NameError', 'AttributeError'))]
    return '\n'.join(user_lines)

def get_error_line_number(tb: str) -> int:
    """Get error line number from traceback - EXACT SAME as original"""
    match = re.search(r'File "<string>", line (\d+)', tb)
    if match:
        return int(match.group(1))
    return -1

# ----------- EXACT SAME CODE EXTRACTION FROM ORIGINAL -----------

def extract_code(text: str) -> str:
    """Extract Python code from model response - EXACT SAME as original"""
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

    # Step 4: Clip after last print() if it exists - FIXED REGEX FROM ORIGINAL
    print_matches = list(re.finditer(r'print\s*\([^()]*(?:\([^()]*\)[^()]*)*\)', code))
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

# ----------- EXACT SAME CODE EXECUTION FROM ORIGINAL -----------

def execute_code(code: str) -> Tuple[str, Optional[str]]:
    """Execute Python code and return output and error - EXACT SAME as original"""
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

# ----------- EXACT SAME PREDICTION NORMALIZATION FROM ORIGINAL -----------

def normalize_prediction(pred: str) -> int:
    """Extract prediction from model response - FIXED VERSION"""
    text = pred.lower()

    # Layer 1: strict pattern match - FIXED REGEX
    match = re.search(r'final\s+answer\s*:?\s*(true|false|unknown)', text)
    if match:
        result = {"true": 1, "false": 0, "unknown": 2}[match.group(1)]
        return result

    # Rest stays the same...
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
    """Extract prediction from code output - EXACT SAME as original"""
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

def load_prompt(file_path: str) -> str:
    """Load prompt from text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Warning: Prompt file {file_path} not found")
        return ""

def print_gpu_info():
    """Print GPU information"""
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            allocated_gb = torch.cuda.memory_allocated(i) / (1024**3)
            
            print(f"GPU {i}: {props.name}")
            print(f"  Total Memory: {memory_gb:.2f} GB")
            print(f"  Allocated Memory: {allocated_gb:.2f} GB")
    else:
        print("CUDA not available")

def log_to_file(content: str, log_file: str):
    """Append content to log file - EXACT SAME as original"""
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(content + '\n')

# ----------- EXACT SAME ANALYSIS FUNCTIONS FROM ORIGINAL -----------

def analyze_agreement(results: List[Dict]) -> Dict:
    """Analyze agreement between reasoning and code predictions - EXACT SAME as original"""
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
        verifier_pred = result.get("verifier_prediction", reasoning_pred)  # Handle missing verifier
        final_pred = result.get("prediction", reasoning_pred)  # Handle missing final prediction
        
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
    """Print detailed analysis of results - EXACT SAME as original"""
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