# Import gpt4all first
from gpt4all import GPT4All

import asyncio
import os
import traceback
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_core.models import CreateResult, RequestUsage

# Optional vector memory (ChromaDB); will be skipped if unavailable
try:
    from autogen_core.memory import MemoryContent, MemoryMimeType
    from autogen_ext.memory.chromadb import (
        ChromaDBVectorMemory,
        PersistentChromaDBVectorMemoryConfig,
        SentenceTransformerEmbeddingFunctionConfig,
    )
    CHROMA_AVAILABLE = True
except Exception:
    CHROMA_AVAILABLE = False

# -------------------
# Model Configuration
# -------------------
MODELS_ROOT = r"C:\Users\Huawei\Desktop\Letta Projects\.venv\models"

# Using Llama for all agents
LLAMA_PATH = os.path.join(MODELS_ROOT, "Llama-3.2-3B-Instruct")
LLAMA_FILE = "Llama-3.2-3B-Instruct-Q4_0.gguf"

INJECTOR_MODEL = {
    "name": LLAMA_FILE,
    "path": LLAMA_PATH
}

DETECTOR_MODEL = {
    "name": LLAMA_FILE,
    "path": LLAMA_PATH
}

# -------------------
# --- PRE-LOAD MODELS SYNCHRONOUSLY ---
# -------------------
print("--- Pre-loading models before starting async operations ---")

injector_model = GPT4All(
    model_name=INJECTOR_MODEL["name"], 
    model_path=INJECTOR_MODEL["path"], 
    allow_download=False
)
print("Injector model loaded.")

detector_model = GPT4All(
    model_name=DETECTOR_MODEL["name"], 
    model_path=DETECTOR_MODEL["path"], 
    allow_download=False
)
print("Detector model loaded.")

print("-" * 50)

# -------------------
# GPT4All client wrapper
# -------------------
class GPT4AllClient:
    def __init__(self, preloaded_model: GPT4All):
        self.model = preloaded_model

    @property
    def model_info(self):
        return {
            "family": "generic",
            "vision": False,
        }

    async def create(self, messages, **kwargs):
        """Create a chat completion response."""
        prompt_parts = []
        system_msg = ""
        
        for msg in messages:
            role = getattr(msg, 'role', 'user')
            content = getattr(msg, 'content', '')
            
            if role == "system":
                system_msg = content
            elif role == "user" or role == "assistant":
                prompt_parts.append(content)
        
        if system_msg:
            prompt = f"{system_msg}\n\n" + "\n\n".join(prompt_parts) + "\n\nResponse:"
        else:
            prompt = "\n\n".join(prompt_parts) + "\n\nResponse:"
        
        # Generate response using GPT4All
        loop = asyncio.get_running_loop()
        response_text = await loop.run_in_executor(
            None, 
            lambda: self.model.generate(prompt, max_tokens=kwargs.get("max_tokens", 150), temp=0.5)
        )
        
        return CreateResult(
            content=response_text.strip(),
            finish_reason="stop",
            usage=RequestUsage(prompt_tokens=0, completion_tokens=0),
            cached=False
        )

# -------------------
# Agent Definitions
# -------------------
def create_injector(model_instance):
    """Creates the FaultInjector agent."""
    injector_system_message = (
        "You are FaultInjector. Create a mutant and test case.\n"
        "RULES:\n"
        "1. Keep the SAME function name\n"
        "2. Change ONLY ONE operator: + to -, * to /, < to >, == to !=, etc.\n"
        "3. Write test case as: function_name(value1, value2)\n"
        "\n"
        "FORMAT (NO extra text):\n"
        "MUTANT:\n"
        "```python\n"
        "[code with ONE operator changed]\n"
        "```\n"
        "TEST: function_name(value1, value2)"
    )
    agent = AssistantAgent(
        name="FaultInjector",
        system_message=injector_system_message,
        model_client=GPT4AllClient(preloaded_model=model_instance)
    )
    return agent

def create_detector(model_instance):
    """Creates the BugDetector agent."""
    detector_system_message = (
        "You are BugDetector. Create a test case to detect the bug.\n"
        "RULES:\n"
        "1. Analyze the mutant code\n"
        "2. Find input values that will expose the bug\n"
        "\n"
        "FORMAT (NO extra text):\n"
        "TEST: function_name(value1, value2)"
    )
    agent = AssistantAgent(
        name="BugDetector",
        system_message=detector_system_message,
        model_client=GPT4AllClient(preloaded_model=model_instance)
    )
    return agent

def maybe_create_chroma_memory(collection_name: str):
    """Create a ChromaDB memory store if dependencies are available; otherwise return None."""
    if not CHROMA_AVAILABLE:
        print(f"ChromaDB not available - running without memory for {collection_name}")
        return None
    try:
        db_path = os.path.join(os.path.expanduser("~"), ".chromadb_autogen")
        memory = ChromaDBVectorMemory(
            config=PersistentChromaDBVectorMemoryConfig(
                collection_name=collection_name,
                persistence_path=db_path,
                k=5,
                score_threshold=0.3,
                embedding_function_config=SentenceTransformerEmbeddingFunctionConfig(
                    model_name="all-MiniLM-L6-v2"
                ),
            )
        )
        print(f"Memory initialized: {collection_name}")
        return memory
    except Exception as e:
        print(f"Memory error for {collection_name}: {str(e)}")
        return None

def extract_code_block(text: str) -> str:
    """Extract code from ```python blocks."""
    if "```python" in text:
        start = text.find("```python") + 9
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()
    return text.strip()

def extract_test_case(text: str) -> str:
    """Extract test case from response."""
    lines = text.split("\n")
    
    # Look for TEST: marker (case insensitive)
    for i, line in enumerate(lines):
        if "TEST:" in line.upper() or "TEST " in line.upper():
            # Check the same line first (after TEST:)
            if ":" in line:
                test_part = line.split(":", 1)[-1].strip()
            else:
                test_part = line.replace("TEST", "", 1).strip()
            
            # Clean up the test case
            test_part = test_part.strip("`").strip("'").strip('"').strip()
            
            # Validate it's a function call, not a definition
            if test_part and "(" in test_part and ")" in test_part:
                if not test_part.startswith("def ") and not test_part.startswith("class "):
                    # Additional check: should not contain "def " at all
                    if "def " not in test_part:
                        return test_part
            
            # Check next line if current line didn't work
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip().strip("`").strip("'").strip('"').strip()
                if next_line and "(" in next_line and ")" in next_line:
                    if not next_line.startswith("def ") and "def " not in next_line:
                        return next_line
    
    return ""

def execute_code(code: str, test_case: str):
    """Execute code with test case and return result."""
    try:
        # Create execution environment
        exec_globals = {}
        
        # Execute the code to define the function
        exec(code, exec_globals)
        
        # Execute the test case
        result = eval(test_case, exec_globals)
        return {"success": True, "result": result, "error": None}
    except Exception as e:
        return {"success": False, "result": None, "error": str(e)}

async def store_round_memory(memory, round_num: int, agent_name: str, content: str):
    """Store memory for an agent."""
    if memory is None:
        return
    try:
        await memory.add(
            MemoryContent(
                content=f"Round {round_num}: {content}",
                mime_type=MemoryMimeType.TEXT,
                metadata={"round": round_num, "agent": agent_name}
            )
        )
    except Exception as e:
        print(f"Failed to store memory for {agent_name}: {str(e)}")

# -------------------
# Main Competition Logic
# -------------------
async def run_competition_round(code_snippet: str, round_num: int):
    """Run one round of bug injection/detection competition."""
    
    # Create agents
    injector = create_injector(injector_model)
    detector = create_detector(detector_model)
    
    # Create separate memories
    injector_memory = maybe_create_chroma_memory("injector_memory")
    detector_memory = maybe_create_chroma_memory("detector_memory")
    
    if injector_memory is not None:
        injector.memory = [injector_memory]
    if detector_memory is not None:
        detector.memory = [detector_memory]
    
    print("\n" + "=" * 60)
    print(f"ROUND {round_num}")
    print("=" * 60)
    print(f"Original Program P:\n```python\n{code_snippet}\n```")
    print("-" * 60)

    injector_score = 0
    detector_score = 0
    
    # PHASE 1: INJECTOR
    print("\n[PHASE 1: INJECTOR]")
    print("Injector creating mutant and test case...")
    
    task = f"Original code:\n```python\n{code_snippet}\n```\n\nCreate a mutant and test case."
    
    termination = MaxMessageTermination(max_messages=2)
    team = RoundRobinGroupChat([injector], termination_condition=termination)
    
    injector_response = ""
    async for message in team.run_stream(task=task):
        if hasattr(message, 'content') and hasattr(message, 'source'):
            if message.source == "FaultInjector":
                injector_response = message.content
                print(f"\n{message.content}")
    
    # Extract mutant and test case
    mutant_code = extract_code_block(injector_response)
    injector_test = extract_test_case(injector_response)
    
    print(f"\n[DEBUG] Extracted Mutant:\n{mutant_code}")
    print(f"[DEBUG] Injector Test Case: {injector_test}")
    
    if not mutant_code or not injector_test:
        print("[ERROR] Failed to extract mutant or test case. Injector gets -1.")
        injector_score = -1
        await store_round_memory(injector_memory, round_num, "Injector", 
                                 f"Failed to create proper mutant/test. Score: {injector_score}")
        
        print(f"\n[FINAL SCORES]")
        print(f"Injector: {injector_score}")
        print(f"Detector: {detector_score}")
        print("=" * 60 + "\n")
        return
    
    # JUDGE: Execute injector's test case on both original and mutant
    print("\n[JUDGE] Executing injector's test case...")
    
    original_result = execute_code(code_snippet, injector_test)
    mutant_result = execute_code(mutant_code, injector_test)
    
    print(f"Original P result: {original_result}")
    print(f"Mutant M result: {mutant_result}")
    
    if not original_result["success"] or not mutant_result["success"]:
        print("[ERROR] Test execution failed. Injector gets -1.")
        injector_score = -1
        await store_round_memory(injector_memory, round_num, "Injector",
                                 f"Test execution failed. Score: {injector_score}")
        
        print(f"\n[FINAL SCORES]")
        print(f"Injector: {injector_score}")
        print(f"Detector: {detector_score}")
        print("=" * 60 + "\n")
        return
    
    if original_result["result"] == mutant_result["result"]:
        print("[JUDGE] Results are EQUAL. Injector failed to create distinguishing test. Injector: -1")
        injector_score = -1
        await store_round_memory(injector_memory, round_num, "Injector",
                                 f"Test did not distinguish mutant. Score: {injector_score}")
        
        print(f"\n[FINAL SCORES]")
        print(f"Injector: {injector_score}")
        print(f"Detector: {detector_score}")
        print("=" * 60 + "\n")
        return
    
    print("[JUDGE] Results DIFFER. Mutant is valid. Forwarding to Detector...")
    
    # PHASE 2: DETECTOR
    print("\n[PHASE 2: DETECTOR]")
    print("Detector analyzing mutant and creating test case...")
    
    detector_task = f"Mutant code:\n```python\n{mutant_code}\n```\n\nCreate a test case to detect the bug."
    
    team2 = RoundRobinGroupChat([detector], termination_condition=MaxMessageTermination(max_messages=2))
    
    detector_response = ""
    async for message in team2.run_stream(task=detector_task):
        if hasattr(message, 'content') and hasattr(message, 'source'):
            if message.source == "BugDetector":
                detector_response = message.content
                print(f"\n{message.content}")
    
    # Extract detector's test case
    detector_test = extract_test_case(detector_response)
    
    print(f"\n[DEBUG] Detector Test Case: {detector_test}")
    
    if not detector_test:
        print("[ERROR] Detector failed to create test case. Detector: -1, Injector: +1")
        detector_score = -1
        injector_score = 1
        
        await store_round_memory(injector_memory, round_num, "Injector",
                                 f"Created valid mutant. Detector failed. Score: {injector_score}")
        await store_round_memory(detector_memory, round_num, "Detector",
                                 f"Failed to create test case. Score: {detector_score}")
        
        print(f"\n[FINAL SCORES]")
        print(f"Injector: {injector_score}")
        print(f"Detector: {detector_score}")
        print("=" * 60 + "\n")
        return
    
    # JUDGE: Execute detector's test case on both original and mutant
    print("\n[JUDGE] Executing detector's test case...")
    
    original_result2 = execute_code(code_snippet, detector_test)
    mutant_result2 = execute_code(mutant_code, detector_test)
    
    print(f"Original P result: {original_result2}")
    print(f"Mutant M result: {mutant_result2}")
    
    if not original_result2["success"] or not mutant_result2["success"]:
        print("[ERROR] Test execution failed. Detector: -1, Injector: +1")
        detector_score = -1
        injector_score = 1
        
        await store_round_memory(injector_memory, round_num, "Injector",
                                 f"Created valid mutant. Detector test failed. Score: {injector_score}")
        await store_round_memory(detector_memory, round_num, "Detector",
                                 f"Test execution failed. Score: {detector_score}")
    elif original_result2["result"] != mutant_result2["result"]:
        print("[JUDGE] Results DIFFER. Detector successfully found the bug! Detector: +1, Injector: -1")
        detector_score = 1
        injector_score = -1
        
        await store_round_memory(injector_memory, round_num, "Injector",
                                 f"Mutant was detected. Score: {injector_score}")
        await store_round_memory(detector_memory, round_num, "Detector",
                                 f"Successfully detected bug. Score: {detector_score}")
    else:
        print("[JUDGE] Results are EQUAL. Detector failed to find bug. Detector: -1, Injector: +1")
        detector_score = -1
        injector_score = 1
        
        await store_round_memory(injector_memory, round_num, "Injector",
                                 f"Mutant survived detection. Score: {injector_score}")
        await store_round_memory(detector_memory, round_num, "Detector",
                                 f"Failed to detect bug. Score: {detector_score}")
    
    print(f"\n[FINAL SCORES]")
    print(f"Injector: {injector_score}")
    print(f"Detector: {detector_score}")
    
    # Close memories
    for memory in [injector_memory, detector_memory]:
        if memory is not None:
            try:
                await memory.close()
            except Exception:
                pass
    
    print("=" * 60 + "\n")


async def main():
    """Run multiple competition rounds with different code snippets."""
    code_snippets = [
        "def add(a, b):\n    return a + b",
        "def multiply(a, b):\n    return a * b",
        "def subtract(a, b):\n    return a - b",
    ]
    
    print("=" * 60)
    print("MULTI-AGENT BUG INJECTION & DETECTION COMPETITION")
    print("=" * 60)
    print(f"Running {len(code_snippets)} rounds with local GPT4All models...")
    print("=" * 60)
    
    for i, snippet in enumerate(code_snippets, start=1):
        await run_competition_round(snippet, i)

if __name__ == "__main__":
    asyncio.run(main())
