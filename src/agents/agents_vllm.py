import textwrap
from typing import Tuple, List, Dict, Optional, Any
from .utils_vllm import (
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
        # Load prompt from file - NO HARDCODED PROMPTS
        self.prompt_template = load_prompt("prompts/reasoner_prompt.txt")
    
    def analyze(self, table: str, statement: str) -> Tuple[str, int]:
        """Analyze table and statement"""
        prompt = self.prompt_template.replace("{table}", table).replace("{statement}", statement)
        response = self.query_model(prompt, max_tokens=1500)
        prediction = normalize_prediction(response)
        return response, prediction

class CoderAgent(BaseAgent):
    """Code generation agent with EXACT auto-debug from original traceback code"""
    
    def __init__(self, llm, enable_auto_debug=True, config=None):
        super().__init__(llm, config or {})
        self.enable_auto_debug = enable_auto_debug
        self.debug_rounds = config.get("debug_rounds", 5) if config else 5  # Default to 5 like original
        
        # Load all prompts from files - NO HARDCODED PROMPTS
        self.base_prompt = load_prompt("prompts/coder_base_prompt.txt")
        self.with_guidance_instructions = load_prompt("prompts/coder_with_guidance.txt")
        self.without_guidance_instructions = load_prompt("prompts/coder_without_guidance.txt")
        self.debug_template = load_prompt("prompts/coder_debug.txt")
    
    def build_code_prompt(self, table: str, statement: str, reasoning_guidance: str = None, 
                         previous_code: str = None, error_msg: str = None) -> str:
        """Build code prompt using loaded templates"""
        
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
        """Generate code with EXACT auto-debug logic from original traceback code"""
        
        # Initial code generation
        prompt = self.build_code_prompt(table, statement, reasoning_guidance)
        raw_code = self.query_model(prompt, max_tokens=2000)
        code = extract_code(raw_code)
        output, error = execute_code(code)
        
        # Initialize debug attempts with initial attempt
        debug_attempts = [{"code": code, "output": output, "error": error, "attempt_type": "initial"}]
        
        # Enhanced auto-debug with detailed logging (EXACT SAME as original)
        if self.enable_auto_debug:
            print(f"Initial Code:\n{textwrap.indent(code, '    ')}", flush=True)
            if error:
                print(f"Initial Error: {error}", flush=True)
            else:
                print("Initial execution succeeded.", flush=True)
            
            for debug_round in range(self.debug_rounds):
                # EXACT SAME logic as original auto-debug code
                invalid_output = (not output.strip()) or (output.strip().lower() == "no data")
                code_invalid = code.startswith("# invalid code")
                
                if not error and not invalid_output and not code_invalid:
                    break  # Only stop if code runs, produces meaningful output, and starts validly
                
                # EXACT SAME detailed logging as original
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
                
                # Generate debug prompt and retry
                error_msg = error or "Output was empty or invalid"
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
            
            # Final logging (EXACT SAME as original)
            print(f"\nFinal Code:\n{textwrap.indent(code, '    ')}", flush=True)
            print(f"Final Output: {output}", flush=True)
            if error:
                print(f"Final Error: {error}", flush=True)
        
        debug_rounds_used = len(debug_attempts) - 1
        
        # Extract prediction from final code output
        if error:
            code_prediction = 2  # Unknown due to error
        else:
            code_prediction = extract_code_prediction(output)
        
        return raw_code, code, output, error, debug_attempts, code_prediction