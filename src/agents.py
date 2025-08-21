import textwrap
from typing import Tuple, List, Dict, Optional, Any
from utils import (
    query_llm, load_prompt, normalize_prediction, 
    extract_code, extract_code_prediction, execute_code, log_to_file
)

class BaseAgent:
    """Base class for all agents"""
    
    def __init__(self, pipe, config: Dict[str, Any]):
        self.pipe = pipe
        self.config = config
        self.max_tokens = config.get("max_tokens", 3000)
    
    def query_model(self, prompt: str, max_tokens: int = None) -> str:
        """Query the model with prompt"""
        max_tokens = max_tokens or self.max_tokens
        return query_llm(self.pipe, prompt, max_tokens)

class PlannerAgent(BaseAgent):
    """Planning agent"""
    
    def __init__(self, pipe, config=None):
        super().__init__(pipe, config or {})
        self.prompt_template = load_prompt("prompts/planner_prompt.txt")
    
    def create_plan(self, table: str, statement: str) -> str:
        """Create reasoning plan"""
        prompt = self.prompt_template.replace("{table}", table).replace("{statement}", statement)
        return self.query_model(prompt, max_tokens=512)

class ReasoningAgent(BaseAgent):
    """Reasoning agent"""
    
    def __init__(self, pipe, config=None):
        super().__init__(pipe, config or {})
        self.prompt_template = load_prompt("prompts/reasoner_prompt.txt")
    
    def analyze(self, table: str, statement: str) -> Tuple[str, int]:
        """Analyze table and statement"""
        prompt = self.prompt_template.replace("{table}", table).replace("{statement}", statement)
        response = self.query_model(prompt, max_tokens=5000)
        prediction = normalize_prediction(response)
        return response, prediction

class CoderAgent(BaseAgent):
    """Code generation agent - EXACTLY same logic as original"""
    
    def __init__(self, pipe, enable_auto_debug=True, config=None):
        super().__init__(pipe, config or {})
        self.enable_auto_debug = enable_auto_debug
        self.debug_rounds = config.get("debug_rounds", 5) if config else 5
        
        # Load ALL prompt templates from txt files
        self.base_prompt = load_prompt("prompts/coder_base_prompt.txt")
        self.with_guidance_instructions = load_prompt("prompts/coder_with_guidance.txt")
        self.without_guidance_instructions = load_prompt("prompts/coder_without_guidance.txt")
        self.debug_template = load_prompt("prompts/coder_debug.txt")
    
    def build_code_prompt(self, table: str, statement: str, reasoning_guidance: str = None, 
                         previous_code: str = None, error_msg: str = None) -> str:
        """Build code prompt - EXACTLY same logic as original"""
        
        # Start with base prompt
        base = self.base_prompt.replace("{table}", table).replace("{statement}", statement)
        
        # Add reasoning guidance if provided - EXACTLY like original
        if reasoning_guidance:
            base += f"\n=== Reasoning Guidance ===\n{reasoning_guidance}\n"
        
        # Different instructions based on whether planner guidance is available - EXACTLY like original
        if reasoning_guidance:
            # With planner - follow the guidance
            base += self.with_guidance_instructions
        else:
            # Without planner - analyze and implement independently
            base += self.without_guidance_instructions
        
        # Add debugging instructions if this is a retry - EXACTLY like original
        if previous_code and error_msg:
            # Apply textwrap.indent to previous_code FIRST (same as original)
            indented_code = textwrap.indent(previous_code, "# ")
            
            # Then replace in debug template (loaded from file)
            debug_section = self.debug_template.replace("{previous_code}", indented_code)
            debug_section = debug_section.replace("{error_msg}", error_msg)
            
            # Add to base with proper newlines (matching original f-string format)
            base += f"\n\n{debug_section}"
        
        return base
    
    def generate_code(self, table: str, statement: str, reasoning_guidance: str = None) -> Tuple[str, str, str, Optional[str], List[Dict], int]:
        """Generate code - EXACTLY same logic as original"""
        prompt = self.build_code_prompt(table, statement, reasoning_guidance)
        raw_code = self.query_model(prompt, max_tokens=5000)
        code = extract_code(raw_code)
        output, error = execute_code(code)
        
        debug_attempts = [{"code": code, "output": output, "error": error}]
        
        # Auto-debug if enabled and there's an error or invalid output - EXACTLY like original
        if self.enable_auto_debug:
            for debug_round in range(self.debug_rounds):
                invalid_output = (not output.strip()) or (output.strip().lower() == "no data")
                code_invalid = code.startswith("# invalid code") or "⚠️ Warning:" in code
                
                if not error and not invalid_output and not code_invalid:
                    break  # Only stop if code runs, produces meaningful output, and starts validly
                
                print(f"Debug Round {debug_round + 1}: Fixing code issues...")
                prompt = self.build_code_prompt(table, statement, reasoning_guidance, code, 
                                              error or "Output was empty or invalid")
                raw_code = self.query_model(prompt, max_tokens=5000)
                code = extract_code(raw_code)
                output, error = execute_code(code)
                debug_attempts.append({"code": code, "output": output, "error": error})
        
        # Extract prediction from code output - EXACTLY like original
        if error:
            code_prediction = 2  # Unknown due to error
        else:
            code_prediction = extract_code_prediction(output)
        
        return raw_code, code, output, error, debug_attempts, code_prediction

class VerifierAgent(BaseAgent):
    """Verifier agent"""
    
    def __init__(self, pipe, config=None):
        super().__init__(pipe, config or {})
        self.prompt_template = load_prompt("prompts/verifier_prompt.txt")
    
    def verify(self, table: str, statement: str, reasoning_output: str, reasoning_prediction: int,
              code: str, output: str, error: Optional[str], code_prediction: int) -> Tuple[str, int]:
        """Verify results"""
        
        # Format code output (handle errors)
        code_output = output if not error else f"Execution Error: {error}"
        
        # Build prompt with replacements
        prompt = self.prompt_template.replace("{table}", table).replace("{statement}", statement)
        prompt = prompt.replace("{reasoning_output}", reasoning_output).replace("{reasoning_prediction}", str(reasoning_prediction))
        prompt = prompt.replace("{code}", code).replace("{code_output}", code_output)
        prompt = prompt.replace("{code_prediction}", str(code_prediction))
        
        response = self.query_model(prompt, max_tokens=1024)
        prediction = normalize_prediction(response)
        
        return response, prediction

class MultiAgentSystem:
    """Main MAS orchestrator - EXACT same logic as original"""
    
    def __init__(self, pipe_coder, pipe_instruct, pipe_r1, enable_auto_debug=True, use_planner=True):
        self.use_planner = use_planner
        if use_planner:
            self.planner_agent = PlannerAgent(pipe_instruct)
        self.reasoning_agent = ReasoningAgent(pipe_r1)
        self.coder_agent = CoderAgent(pipe_coder, enable_auto_debug)
        self.verifier_agent = VerifierAgent(pipe_instruct)
    
    def process_sample(self, table: str, statement: str, label: str, log_file: str) -> Dict:
        """Process single sample - EXACT same logic as original"""
        
        log_content = f"\n{'='*80}\nSample Processing\n{'='*80}\n"
        log_content += f"Statement: {statement}\n"
        log_content += f"Label: {label}\n"
        log_content += f"Table: {table}\n\n"
        
        reasoning_guidance = None
        
        if self.use_planner:
            print("Step 1: Planner Agent creating reasoning plan...")
            log_content += "STEP 1: PLANNER AGENT\n" + "-"*40 + "\n"
            reasoning_guidance = self.planner_agent.create_plan(table, statement)
            log_content += f"Reasoning Guidance:\n{reasoning_guidance}\n\n"
        
        step_num = 2 if self.use_planner else 1
        print(f"Step {step_num}: Reasoning Agent analyzing...")
        log_content += f"STEP {step_num}: REASONING AGENT\n" + "-"*40 + "\n"
        reasoning_output, reasoning_prediction = self.reasoning_agent.analyze(table, statement)
        log_content += f"Reasoning Output:\n{reasoning_output}\n"
        log_content += f"Reasoning Prediction: {reasoning_prediction}\n\n"
        
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
        
        step_num = 4 if self.use_planner else 3
        print(f"Step {step_num}: Verifier Agent making final conclusion...")
        log_content += f"STEP {step_num}: VERIFIER AGENT\n" + "-"*40 + "\n"
        verifier_output, verifier_prediction = self.verifier_agent.verify(
            table, statement, reasoning_output, reasoning_prediction,
            code, code_output, code_error, code_prediction
        )
        log_content += f"Verifier Output:\n{verifier_output}\n"
        log_content += f"Verifier Prediction: {verifier_prediction}\n\n"
        
        label_int = int(label)
        reasoning_correct = (reasoning_prediction == label_int)
        code_correct = (code_prediction == label_int)
        verifier_correct = (verifier_prediction == label_int)
        methods_agree = (reasoning_prediction == code_prediction)
        
        final_prediction = verifier_prediction
        final_correct = verifier_correct
        
        log_content += "RESULTS\n" + "-"*40 + "\n"
        log_content += f"Reasoning Correct: {reasoning_correct}\n"
        log_content += f"Code Correct: {code_correct}\n"
        log_content += f"Verifier Correct: {verifier_correct}\n"
        log_content += f"Methods Agree: {methods_agree}\n"
        log_content += f"Final Prediction: {final_prediction}\n"
        log_content += f"Final Correct: {final_correct}\n"
        
        log_to_file(log_content, log_file)
        
        result = {
            "sample_id": 0,
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