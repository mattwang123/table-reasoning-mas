import json
import io
import sys
import re
import warnings
import textwrap
import torch
import pandas as pd
import gc
from typing import List, Dict, Tuple, Optional, Any
from transformers import pipeline

# ============================================================================
# MODEL UTILITIES
# ============================================================================

def print_gpu_info():
    """Print GPU information"""
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        print(f"  Allocated Memory: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
        print(f"  Cached Memory: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")

def clear_gpu_memory():
    """Clear GPU memory"""
    torch.cuda.empty_cache()
    gc.collect()

def load_models(config: Dict[str, Any]) -> Dict[str, Any]:
    """Load all models based on configuration"""
    models = {}
    
    model_configs = {
        "coder": config["models"]["coder"],
        "instruct": config["models"]["instruct"], 
        "reasoning": config["models"]["reasoning"]
    }
    
    for model_name, model_path in model_configs.items():
        print(f"Loading {model_name} model: {model_path}")
        models[f"pipe_{model_name}"] = pipeline(
            "text-generation", 
            model=model_path, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
    
    return models

def query_llm(pipe, prompt: str, max_tokens: int = 3000) -> str:
    """Query language model with error handling"""
    try:
        response = pipe(prompt, max_new_tokens=max_tokens)[0]["generated_text"]
        result = response[len(prompt):].strip()
        return result
    except Exception as e:
        return f"# Generation error: {e}"

# ============================================================================
# TEXT PROCESSING UTILITIES
# ============================================================================

def extract_code(text: str) -> str:
    """Extract and clean code from LLM response - EXACT same logic as original"""
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
        if code.count('(') != code.count(')'):
            return False
        if code.count('[') != code.count(']'):
            return False
        if code.count('{') != code.count('}'):
            return False
        if re.search(r'\bdef\b.*[^:]\n', code):
            return False
        return True

    valid_starts = ('import', 'from', 'def', 'class', '#', 'data =', 'df =', 'table =', 'try:', 'pd.')
    if not code.startswith(valid_starts) or not looks_complete(code):
        print("Warning: Extracted code may be invalid, incomplete, or poorly formatted")

    return code

def normalize_prediction(pred: str) -> int:
    """Normalize prediction to 0/1/2 - EXACT same logic as original"""
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

    return 2

def extract_code_prediction(code_output: str) -> int:
    """Extract prediction from code output - EXACT same logic as original"""
    if not code_output or code_output.strip() == "No output":
        return 2
    
    text = code_output.lower()
    
    if "true" in text and "false" not in text:
        return 1
    elif "false" in text and "true" not in text:
        return 0
    
    lines = code_output.strip().split('\n')
    for line in lines:
        line = line.strip().lower()
        if 'true' in line:
            return 1
        elif 'false' in line:
            return 0
    
    return 2

def load_prompt(prompt_file: str) -> str:
    """Load prompt from text file"""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read().strip()

# ============================================================================
# CODE EXECUTION
# ============================================================================

def execute_code(code: str) -> Tuple[str, str]:
    """Execute code and capture output - EXACT same logic as original"""
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

# ============================================================================
# ANALYSIS UTILITIES
# ============================================================================

def analyze_agreement(results: List[Dict]) -> Dict:
    """Analyze agreement between reasoning and code predictions - EXACT same logic as original"""
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
        else:
            if reasoning_correct:
                stats["disagree_reasoning_correct"] += 1
            elif code_correct:
                stats["disagree_code_correct"] += 1
    
    if stats["total_samples"] > 0:
        stats["reasoning_accuracy"] = stats["reasoning_correct"] / stats["total_samples"]
        stats["code_accuracy"] = stats["code_correct"] / stats["total_samples"] 
        stats["verifier_accuracy"] = stats["verifier_correct"] / stats["total_samples"]
        stats["final_accuracy"] = stats["final_correct"] / stats["total_samples"]
        
        agreement_count = sum(1 for r in results if r["reasoning_prediction"] == r["code_prediction"])
        stats["agreement_rate"] = agreement_count / stats["total_samples"]
    
    return stats

def print_analysis(stats: Dict, log_file: str = None):
    """Print detailed analysis of results - EXACT same logic as original"""
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

# ============================================================================
# I/O UTILITIES
# ============================================================================

def append_jsonl(result: Dict, file_path: str):
    """Append result to JSONL file"""
    with open(file_path, 'a', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)
        f.write('\n')

def log_to_file(content: str, log_file: str):
    """Append content to log file"""
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(content + '\n')