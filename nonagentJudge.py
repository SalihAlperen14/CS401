# agent_competition.py
import asyncio
import os
from dotenv import load_dotenv
from autogen_core.memory import Memory, MemoryContent, MemoryMimeType
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent, CodeExecutorAgent
from autogen_agentchat.teams import SelectorGroupChat, Swarm
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.ui import Console
from typing import Any, Dict, List
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
import chromadb
from modifiedAgents import LastMessageDefenderAgent
from autogen_ext.memory.chromadb import (
    ChromaDBVectorMemory,
    PersistentChromaDBVectorMemoryConfig,
    SentenceTransformerEmbeddingFunctionConfig,
)
import random, re
import os
import subprocess
import textwrap
import shlex
import os
import subprocess
from langchain_core.documents import Document

from typing import Dict, Optional
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

model_client = OpenAIChatCompletionClient(
    model="qwen2.5-coder:14b",          # The exact model name you pulled in Ollama
    base_url="http://localhost:11434/v1", # Point to local Ollama server
    api_key="ollama",               # Ollama requires an API key argument, but it can be any string
    model_info={                    # Optional: Helps AutoGen understand local model capabilities
        "vision": False,
        "function_calling": False,
        "json_output": False,
        "family": "unknown",
    },
) 

docker_executor = DockerCommandLineCodeExecutor(work_dir="coding")
docker_tool = PythonCodeExecutionTool(docker_executor)

def remove_comments(code):
    """
    Removes inline and block comments (#) from Python code 
    while preserving strings and docstrings.
    """
    # Pattern explanation:
    # Group 1 matches strings: triple-quoted, double-quoted (with escapes), single-quoted (with escapes)
    # Group 2 matches comments: starts with # and goes to end of line
    pattern = r"(\"\"\"[\s\S]*?\"\"\"|\'\'\'[\s\S]*?\'\'\'|\"(?:\\.|[^\"\\])*\"|\'(?:\\.|[^'\\])*\')|(#.*)"
    
    def replacer(match):
        # If Group 2 (comment) is found, replace with empty string
        if match.group(2):
            return ""
        # Otherwise it's a string (Group 1), return it exactly as is
        return match.group(1)
        
    return re.sub(pattern, replacer, code)

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
attacker_agent = AssistantAgent(
    name="AttackerAgent",
    description="Attack the code to insert bug.",
    model_client=model_client,
    reflect_on_tool_use = True,
    system_message=(
        """
        You are responsible for inserting a bug into the code you are given. 
        Insert subtle, non-syntax bugs so the code still runs but produces incorrect behavior. 
        Don't reveal where the bug is — just return the modified code.
        Don't add any explanations or comments even in python script. Dont add any comments in the code even like #bug inserted here.
        After you have inserted the bug, generate a single test case (not in the Fixed Code's script) where it also print without any explanation to make sure you inserted bug.
        At the end, return in following format:
        Modified Code:
        <modified code here>

        Test Case:
        <test case here>
        """
    ),
)

defender_agent = AssistantAgent(
    name="DefenderAgent",
    description="Fix the code in case there is a bug.",
    model_client=model_client,
    reflect_on_tool_use = True,
    system_message=(
        """
        You are responsible for fixing the given code if there are bugs.
        Don't reveal where the bug is — just return the modified code.
        Don't add any explanations or comments even in python script.
        After you have fixed the bug, generate a single test case (not in the Fixed Code's script) where it also print without any explanation to make sure you fixed bug.
        At the end, return in following format:

        Fixed Code:
        <fixed code here>

        Test Case:
        <test case here>
        """
    ),
)
def list_function_files(folder_path="Functions"):
    return [f for f in os.listdir(folder_path) if f.endswith(".py")]

def read_function_file(filepath):
    with open(filepath, "r") as f:
        return f.read().strip()

async def main() -> None:
    folder_path = "functions"
    folder_path = "Functions_nodescription"
    function_files = list_function_files(folder_path)

    if not function_files:
        print("No function files found under 'functions/'")
        return

    # 🔁 For each round, pick one function to compete on
    attacker_agent_score = 0
    defender_agent_score = 0
    number_of_rounds = 5
    main_code_file = os.path.join("coding", "main_code.py")
    modified_file_attacker = os.path.join("coding", "attacker_modified.py")
    test_runner_file_attacker = os.path.join("coding", "test_runner_attacker.py")
    test_runner_file_judge = os.path.join("coding", "test_runner_judge.py")
    modified_file_defender = os.path.join("coding", "defender_modified.py")
    test_runner_file_defender = os.path.join("coding", "test_runner_defender.py") 

    #function_files = function_files[:25]  # Limit to number_of_rounds files for testing
    for i,function_file in enumerate(function_files):
        print(f"--- Round {i+1}: {str(function_file)} ---")

        # Pick a random function file (or sequentially if you prefer)
        func_path = os.path.join(folder_path, function_file)
        function_code = read_function_file(func_path)
        # Create the round-specific task
        attacker_task = f"""    
Can you add a bug in the following code?:
{function_code}
"""

# --- Start of the fix ---
        attacker_full_response = ""
        defender_full_response = ""
        
            # Asynchronously iterate over the streaming output
        async for item in attacker_agent.run_stream(task=attacker_task):
                # Check for message content and append it
                if hasattr(item, 'content') and isinstance(item.content, str):
                    attacker_full_response += item.content
                    
                # Note: The 'item' can be a message or an event.
                # If you want to see tool calls, you would check the type of 'item' here.
        print(attacker_full_response)
        modified_code_attacker = re.search(r"Modified Code:\s*```python\n(.*?)```", attacker_full_response, re.DOTALL).group(1).strip()
        test_case_attacker = re.search(r"Test Case:\s*```python\n(.*?)```", attacker_full_response, re.DOTALL).group(1).strip()
        
        with open(modified_file_attacker, "w", encoding="utf-8") as f:
            f.write(modified_code_attacker.rstrip() + "\n")
        with open(main_code_file, "w", encoding="utf-8") as f:
            f.write(function_code.rstrip() + "\n")
        with open(test_runner_file_attacker, "w", encoding="utf-8") as f:
            f.write("from attacker_modified import *\n\n")
            f.write(test_case_attacker + "\n\n")
        with open(test_runner_file_judge, "w", encoding="utf-8") as f:
            f.write("from main_code import *\n\n")
            f.write(test_case_attacker + "\n\n")
        
        
        try:
            res_attacker = docker_runner("coding/test_runner_attacker.py", timeout=10)
        except subprocess.TimeoutExpired:
            print("Attacker's test case execution timed out.")
            res_attacker = {"stdout": "Timeout", "stderr": "Timeout", "returncode": -1}

        res_judge = docker_runner("coding/test_runner_judge.py")
        
        print("Attacker's Output:")
        print(res_attacker["stdout"])
        print("Expected Output:")
        print(res_judge["stdout"])
        modified_code_attacker = remove_comments(modified_code_attacker)
        if res_attacker["stdout"] != res_judge["stdout"]: # If outputs differ, attacker succeeded
            attacker_agent_score += 1
            defender_task= f"""
            Here is the modified code with a potential bug:
    {modified_code_attacker}
            Please fix any bugs you find in the code above.
    """
            async for item in defender_agent.run_stream(task=defender_task):
                    # Check for message content and append it
                    if hasattr(item, 'content') and isinstance(item.content, str):
                        defender_full_response += item.content
            print(defender_full_response)
            modified_code_defender = re.search(r"Fixed Code:\s*```python\n(.*?)```", defender_full_response, re.DOTALL).group(1).strip()
            test_case_defender = re.search(r"Test Case:\s*```python\n(.*?)```", defender_full_response, re.DOTALL).group(1).strip()
            
            with open(modified_file_defender, "w", encoding="utf-8") as f:
                f.write(modified_code_defender.rstrip() + "\n")
            with open(test_runner_file_defender, "w", encoding="utf-8") as f:
                f.write("from defender_modified import *\n\n")
                f.write(test_case_defender + "\n\n")
            with open(test_runner_file_judge, "w", encoding="utf-8") as f:
                f.write("from main_code import *\n\n")
                f.write(test_case_defender + "\n\n")
            
            res_defender = docker_runner("coding/test_runner_defender.py")
            res_judge = docker_runner("coding/test_runner_judge.py")
            
            print("Defender's Output:")
            print(res_defender["stdout"])
            print("Expected Output:")
            print(res_judge["stdout"])
            #filtered_content = last_message_content.replace("TERMINATE", "").strip()
            if res_defender["stdout"] == res_judge["stdout"]: # If outputs match, defender succeeded
                defender_agent_score += 1
                attacker_agent_score -= 1  # Penalize attacker for failing to insert a lasting bug
                history = f"""
                Attacker inserted a bug that changes output. However, Defender succedded to find it.
                Function Code:
                {function_code}
                Modified Code:
                {modified_code_attacker}
                Fixed Code:
                {modified_code_defender}
"""
                
            else:
                defender_agent_score -= 1  # Penalize defender for failing to fix the bug
                history = f"""
                Attacker inserted a bug that changes output, and Defender failed to fix it.
                Function Code:
                {function_code}
                Modified Code:
                {modified_code_attacker}
                Fixed Code:
                {modified_code_defender}
"""

        else:
            history = f"""
            Attacker failed to insert a bug that changes output.
            Function Code:
            {function_code}
            Modified Code:
            {modified_code_attacker}
"""

            attacker_agent_score -= 1  # Penalize attacker for failing to insert a bug

        
        print(f"Scores after Round {i+1}: Attacker: {attacker_agent_score}, Defender: {defender_agent_score}\n")


if __name__ == "__main__":
    asyncio.run(main())

