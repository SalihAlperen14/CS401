import asyncio
import os
import re
import subprocess
from typing import Dict, Optional, Tuple
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

# --- Configuration ---
WORK_DIR = "coding"
os.makedirs(WORK_DIR, exist_ok=True)

model_client = OpenAIChatCompletionClient(
    model="qwen2.5-coder",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    model_info={
        "vision": False,
        "function_calling": False,
        "json_output": False,
        "family": "unknown",
    },
)

# --- Helper Functions ---

def parse_llm_response(response: str) -> Tuple[str, str]:
    """
    Robustly extracts code and test cases. 
    Returns (code, test_case) or raises ValueError if parsing fails.
    """
    # Try specific headers first
    code_match = re.search(r"Modified Code:.*?```python\n(.*?)```", response, re.DOTALL)
    if not code_match:
         # Fallback: Try Fixed Code header
        code_match = re.search(r"Fixed Code:.*?```python\n(.*?)```", response, re.DOTALL)
    
    test_match = re.search(r"Test Case:.*?```python\n(.*?)```", response, re.DOTALL)

    if not code_match or not test_match:
        # Emergency Fallback: Just look for the first two python blocks
        blocks = re.findall(r"```python\n(.*?)```", response, re.DOTALL)
        if len(blocks) >= 2:
            return blocks[0].strip(), blocks[1].strip()
        else:
            raise ValueError("Could not parse Code and Test Case from LLM response.")

    return code_match.group(1).strip(), test_match.group(1).strip()
def docker_runner(script_path: str,
                  image: str = "python:3.11-slim",
                  workdir_in_container: str = "/app",
                  timeout: Optional[int] = None) -> Dict[str, object]:

    # Normalize and check script path
    host_script = os.path.abspath(script_path)
    if not os.path.isfile(host_script):
        raise FileNotFoundError(f"Script not found: {host_script}")

    host_dir = os.path.dirname(host_script)
    script_basename = os.path.basename(host_script)

    # Build docker run command
    # -v host_dir:workdir_in_container to mount the directory
    # --rm to remove container afterwards
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{host_dir}:{workdir_in_container}",
        "-w", workdir_in_container,
        image,
        "python", script_basename
    ]

    # Run and capture output
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    return {
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "returncode": proc.returncode
    }
# --- Agents ---

attacker_agent = AssistantAgent(
    name="AttackerAgent",
    model_client=model_client,
    system_message="""
    You are an expert software tester. Your goal is to insert a SUBTLE LOGIC BUG into code.
    1. The code must still run without syntax errors.
    2. Do NOT add comments or explanations.
    3. Provide the output in the exact requested format.
    """
)

defender_agent = AssistantAgent(
    name="DefenderAgent",
    model_client=model_client,
    system_message="""
    You are an expert software engineer. Your goal is to fix bugs in the provided code.
    1. Restore correct functionality.
    2. Do NOT add comments or explanations.
    3. Provide the output in the exact requested format.
    """
)

# --- Main Logic ---

async def run_round(function_file: str, function_code: str, round_idx: int) -> Tuple[int, int]:
    """Runs a single round. Returns (attacker_score, defender_score)."""
    
    print(f"\n=== Round {round_idx}: {function_file} ===")
    
    # 1. ATTACK PHASE
    attacker_task = f"""    
    Add a subtle logic bug to this code.
    Return format:
    Modified Code:
    ```python
    ...
    ```
    Test Case (a print statement demonstrating the bug):
    ```python
    ...
    ```
    Code to modify:
    {function_code}
    """
    
    print(">> Attacker is generating bug...")
    attacker_response = ""
    async for item in attacker_agent.run_stream(task=attacker_task):
        if hasattr(item, 'content') and isinstance(item.content, str):
            attacker_response += item.content

    try:
        mod_code, test_case = parse_llm_response(attacker_response)
    except ValueError:
        print("!! Parsing Error on Attacker Response. Round Void.")
        return 0, 0

    # Write files
    with open(f"{WORK_DIR}/attacker.py", "w") as f: f.write(mod_code)
    with open(f"{WORK_DIR}/original.py", "w") as f: f.write(function_code)
    
    # Create runners that import specific files
    runner_code = f"try:\n    from {{module}} import *\n    {test_case}\nexcept Exception as e:\n    print(e)"
    
    with open(f"{WORK_DIR}/run_attack.py", "w") as f: f.write(runner_code.format(module="attacker"))
    with open(f"{WORK_DIR}/run_judge.py", "w") as f: f.write(runner_code.format(module="original"))
    print(attacker_response)
    # Execute
    res_att = docker_runner(f"{WORK_DIR}/run_attack.py")
    res_judge = docker_runner(f"{WORK_DIR}/run_judge.py")
    print(res_att)
    print(res_judge)
    print(f"   Judge Output: {res_judge['stdout']}")
    print(f"   Attack Output: {res_att['stdout']}")

    # Validation: Did the attacker confuse the code?
    # CRITICAL CHECK: Attacker only wins if returncode is 0 (valid syntax) AND output is different
    if res_att['returncode'] != 0:
        print("!! Attacker failed: Code crashed (Syntax Error or Runtime Exception).")
        return -1, 0 # Penalty for writing bad code
    
    if res_att['stdout'] == res_judge['stdout']:
        print("!! Attacker failed: Output matches original (Bug ineffective).")
        return -1, 0

    print(">> Attack Successful! Deploying Defender...")
    
    # 2. DEFENSE PHASE
    defender_task = f"""
    The following code has a bug. Fix it.
    Fixed Code:
    ```python
    ...
    ```
    Test Case:
    ```python
    {test_case}
    ```
    Buggy Code:
    {mod_code}
    """

    defender_response = ""
    async for item in defender_agent.run_stream(task=defender_task):
        if hasattr(item, 'content') and isinstance(item.content, str):
            defender_response += item.content
            
    try:
        fixed_code, _ = parse_llm_response(defender_response)
    except ValueError:
        print("!! Parsing Error on Defender Response.")
        return 1, -1

    with open(f"{WORK_DIR}/defender.py", "w") as f: f.write(fixed_code)
    with open(f"{WORK_DIR}/run_defend.py", "w") as f: f.write(runner_code.format(module="defender"))

    res_def = docker_runner(f"{WORK_DIR}/run_defend.py")
    print(f"   Defense Output: {res_def['stdout']}")

    if res_def['stdout'] == res_judge['stdout']:
        print(">> Defender Success! Logic restored.")
        return 0, 1 # Attacker loses their point (or stays neutral), Defender gains
    else:
        print("!! Defender Failed: Output still incorrect.")
        return 1, -1 # Attacker keeps point, Defender loses

async def main():
    # Mocking reading files for demonstration
    # In real usage: function_files = os.listdir("Functions")
    function_files = ["sample_func.py"] 
    # Create a dummy file if it doesn't exist for testing
    if not os.path.exists("Functions_nodescription"):
         os.makedirs("Functions_nodescription", exist_ok=True)
         with open("Functions_nodescription/sample_func.py", "w") as f:
             f.write("def add(a, b):\n    return a + b")

    score_att = 0
    score_def = 0

    folder = "Functions_nodescription"
    files = [f for f in os.listdir(folder) if f.endswith(".py")]

    for i, file in enumerate(files):
        with open(os.path.join(folder, file), "r") as f:
            code = f.read()
        
        sa, sd = await run_round(file, code, i+1)
        score_att += sa
        score_def += sd
        print(f"--- Current Score | Attacker: {score_att} | Defender: {score_def} ---\n")

if __name__ == "__main__":
    asyncio.run(main())