import textwrap
from typing import Tuple, List, Dict, Optional, Any
from utils_vllm import (
    query_llm_vllm, load_prompt, normalize_prediction, 
    extract_code, extract_code_prediction, execute_code
)

class BaseAgent:
    """Base class for all agents"""
    
    def __init__(self, llm, config: Dict[str, Any]):
        self.llm = llm
        self.config = config
        self.max_tokens = config.get("max_tokens", 3000)
    
    def query_model(self, prompt: str, max_tokens: int = None) -> str:
        """Query the VLLM model"""
        max_tokens = max_tokens or self.max_tokens
        return query_llm_vllm(self.llm, prompt, max_tokens)

class ReasoningAgent(BaseAgent):
    """Reasoning agent"""
    
    def __init__(self, llm, config=None):
        super().__init__(llm, config or {})
        self.prompt_template = load_prompt("prompts/reasoner_prompt.txt")
    
    def analyze(self, table: str, statement: str) -> Tuple[str, int]:
        """Analyze table and statement"""
        prompt = self.prompt_template.replace("{table}", table).replace("{statement}", statement)
        response = self.query_model(prompt, max_tokens=1500)
        prediction = normalize_prediction(response)
        return response, prediction

class CoderAgent(BaseAgent):
    """Code generation agent"""
    
    def __init__(self, llm, enable_auto_debug=True, config=None):
        super().__init__(llm, config or {})
        self.enable_auto_debug = enable_auto_debug
        self.debug_rounds = config.get("debug_rounds", 3) if config else 3
        
        self.base_prompt = load_prompt("prompts/coder_base_prompt.txt")
        self.with_guidance_instructions = load_prompt("prompts/coder_with_guidance.txt")
        self.without_guidance_instructions = load_prompt("prompts/coder_without_guidance.txt")
        self.debug_template = load_prompt("prompts/coder_debug.txt")
    
    def build_code_prompt(self, table: str, statement: str, reasoning_guidance: str = None, 
                         previous_code: str = None, error_msg: str = None) -> str:
        """Build code prompt"""
        
        base = self.base_prompt.replace("{table}", table).replace("{statement}", statement)
        
        if reasoning_guidance:
            base += f"\n=== Reasoning Guidance ===\n{reasoning_guidance}\n"
        
        if reasoning_guidance:
            base += self.with_guidance_instructions
        else:
            base += self.without_guidance_instructions
        
        if previous_code and error_msg:
            indented_code = textwrap.indent(previous_code, "# ")
            debug_section = self.debug_template.replace("{previous_code}", indented_code)
            debug_section = debug_section.replace("{error_msg}", error_msg)
            base += f"\n\n{debug_section}"
        
        return base
    
    def generate_code(self, table: str, statement: str, reasoning_guidance: str = None) -> Tuple[str, str, str, Optional[str], List[Dict], int]:
        """Generate code with debug counting"""
        
        prompt = self.build_code_prompt(table, statement, reasoning_guidance)
        raw_code = self.query_model(prompt, max_tokens=2000)
        code = extract_code(raw_code)
        output, error = execute_code(code)
        
        debug_attempts = []
        debug_attempts.append({"code": code, "output": output, "error": error, "attempt_type": "initial"})
        
        if self.enable_auto_debug:
            for debug_round in range(self.debug_rounds):
                invalid_output = (not output.strip()) or (output.strip().lower() == "no data")
                code_invalid = code.startswith("# invalid code") or "Warning:" in code
                generation_error = "generation error" in output.lower() if output else False
                
                if not error and not invalid_output and not code_invalid and not generation_error:
                    break
                
                error_msg = error or "Output was empty, invalid, or contained generation errors"
                prompt = self.build_code_prompt(table, statement, reasoning_guidance, code, error_msg)
                
                raw_code = self.query_model(prompt, max_tokens=2000)
                code = extract_code(raw_code)
                output, error = execute_code(code)
                
                debug_attempts.append({
                    "code": code, 
                    "output": output, 
                    "error": error, 
                    "attempt_type": f"debug_round_{debug_round + 1}"
                })
        
        debug_rounds_used = len(debug_attempts) - 1
        
        if error:
            code_prediction = 2
        else:
            code_prediction = extract_code_prediction(output)
        
        return raw_code, code, output, error, debug_attempts, code_prediction