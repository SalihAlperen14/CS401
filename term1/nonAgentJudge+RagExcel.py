# agent_competition.py
import asyncio
import os
from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
import random, re
import subprocess
from typing import Dict, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import pandas as pd # <--- NEW IMPORT

load_dotenv()
my_api_key = os.getenv("OPENAI_API_KEY")
if not my_api_key:
    raise RuntimeError("OPENAI_API_KEY not found in environment. Put it in .env or export it.")

model_client = OpenAIChatCompletionClient(
    model="gpt-4o-2024-08-06",
    api_key=my_api_key,
)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=my_api_key)

# Initialize ChromaDB as Vector Store
vector_store = Chroma(
    collection_name="CS401",
    embedding_function=embeddings,
    # persist_directory="./chroma_db" # Optional: if you want to save DB to disk
)
docker_executor = DockerCommandLineCodeExecutor(work_dir="coding")
docker_tool = PythonCodeExecutionTool(docker_executor)

def add_history(vector_store, history):
    vector_store.add_documents([history])

def retrieve_history(vector_store, query, k=5):
    return vector_store.similarity_search(query, k=k)

def clear_history(vector_store):
    vector_store.delete_collection()

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
        Don't add any explanations or comments even in python script.
        After you have inserted the bug, generate a single test case (not in the Fixed Code's script) where it also print without any explanation to make sure you inserted bug.
        At the end, return in following format:
        Modified Code:
        ```python
        <modified code here>
        ```

        Test Case:
        ```python
        <test case here>
        ```
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
        ```python
        <fixed code here>
        ```

        Test Case:
        ```python
        <test case here>
        ```
        """
    ),
)

def list_function_files(folder_path="Functions"):
    return [f for f in os.listdir(folder_path) if f.endswith(".py")]

def read_function_file(filepath):
    with open(filepath, "r") as f:
        return f.read().strip()

async def main() -> None:
    # Ensure coding directory exists
    if not os.path.exists("coding"):
        os.makedirs("coding")

    folder_path = "Functions_nodescription"
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    function_files = list_function_files(folder_path)

    if not function_files:
        print(f"No function files found under '{folder_path}'")
        return

    # 🔁 For each round, pick one function to compete on
    attacker_agent_score = 0
    defender_agent_score = 0
    number_of_rounds = 5
    
    # Paths
    main_code_file = os.path.join("coding", "main_code.py")
    modified_file_attacker = os.path.join("coding", "attacker_modified.py")
    test_runner_file_attacker = os.path.join("coding", "test_runner_attacker.py")
    test_runner_file_judge = os.path.join("coding", "test_runner_judge.py")
    modified_file_defender = os.path.join("coding", "defender_modified.py")
    test_runner_file_defender = os.path.join("coding", "test_runner_defender.py") 

    # List to collect data for Excel
    results_data = []
    function_files = function_files[:24]  # Limit to number_of_rounds files
    for i,func_file in enumerate(function_files):
        print(f"--- Round {i+1}: {str(func_file)} ---")

        # Pick a random function file
        func_path = os.path.join(folder_path, func_file)
        function_code = read_function_file(func_path)
        
        # Retrieval
        closest_functions = retrieve_history(vector_store, f"""I want to have the most similar code case for the following code:
        {function_code}""", k=3)
        closest_functions_texts = "\n".join([doc.page_content for doc in closest_functions])
        
        # Initialize Data Dictionary for this Round
        round_data = {
            "Round": i + 1,
            "File Name": func_file,
            "RAG Context Used": closest_functions_texts[:500] + "..." if len(closest_functions_texts) > 500 else closest_functions_texts, # Truncate for excel readability
            "Attacker Success": "No",
            "Defender Success": "N/A",
            "Attacker Output": "",
            "Expected Output (Attacker Phase)": "",
            "Defender Output": "N/A",
            "Expected Output (Defender Phase)": "N/A",
            "History Added to DB": ""
        }

        # Create the round-specific task
        attacker_task = f"""    
Can you add a bug in the following code?:
{function_code}
These are some similar code modification histories that might help you:
{closest_functions_texts}
"""

        attacker_full_response = ""
        defender_full_response = ""
        
        # Run Attacker
        async for item in attacker_agent.run_stream(task=attacker_task):
            if hasattr(item, 'content') and isinstance(item.content, str):
                attacker_full_response += item.content
        
        print(attacker_full_response)
        
        try:
            modified_code_attacker = re.search(r"Modified Code:\s*```python\n(.*?)```", attacker_full_response, re.DOTALL).group(1).strip()
            test_case_attacker = re.search(r"Test Case:\s*```python\n(.*?)```", attacker_full_response, re.DOTALL).group(1).strip()
        except AttributeError:
            print("Error parsing Attacker response. Skipping round.")
            round_data["Attacker Success"] = "Error Parsing"
            results_data.append(round_data)
            continue
        
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
        
        res_attacker = docker_runner("coding/test_runner_attacker.py")
        res_judge = docker_runner("coding/test_runner_judge.py")
        
        print("Attacker's Output:")
        print(res_attacker["stdout"])
        print("Expected Output:")
        print(res_judge["stdout"])
        
        round_data["Attacker Output"] = res_attacker["stdout"].strip()
        round_data["Expected Output (Attacker Phase)"] = res_judge["stdout"].strip()
        
        if res_attacker["stdout"] != res_judge["stdout"]: # If outputs differ, attacker succeeded
            attacker_agent_score += 1
            round_data["Attacker Success"] = "Yes"
            
            defender_task= f"""
            Here is the modified code with a potential bug:
            {modified_code_attacker}
            Please fix any bugs you find in the code above.
            These are some similar code modification histories that might help you:
            {closest_functions_texts}
            """
            async for item in defender_agent.run_stream(task=defender_task):
                if hasattr(item, 'content') and isinstance(item.content, str):
                    defender_full_response += item.content
            print(defender_full_response)
            
            try:
                modified_code_defender = re.search(r"Fixed Code:\s*```python\n(.*?)```", defender_full_response, re.DOTALL).group(1).strip()
                test_case_defender = re.search(r"Test Case:\s*```python\n(.*?)```", defender_full_response, re.DOTALL).group(1).strip()
            except AttributeError:
                print("Error parsing Defender response.")
                round_data["Defender Success"] = "Error Parsing"
                defender_agent_score -= 1
            else:
                with open(modified_file_defender, "w", encoding="utf-8") as f:
                    f.write(modified_code_defender.rstrip() + "\n")
                with open(test_runner_file_defender, "w", encoding="utf-8") as f:
                    f.write("from defender_modified import *\n\n")
                    f.write(test_case_defender + "\n\n")
                with open(test_runner_file_judge, "w", encoding="utf-8") as f:
                    f.write("from main_code import *\n\n")
                    f.write(test_case_defender + "\n\n")
                
                res_defender = docker_runner("coding/test_runner_defender.py")
                res_judge_def = docker_runner("coding/test_runner_judge.py")
                
                print("Defender's Output:")
                print(res_defender["stdout"])
                print("Expected Output:")
                print(res_judge_def["stdout"])
                
                round_data["Defender Output"] = res_defender["stdout"].strip()
                round_data["Expected Output (Defender Phase)"] = res_judge_def["stdout"].strip()

                if res_defender["stdout"] == res_judge_def["stdout"]: # If outputs match, defender succeeded
                    defender_agent_score += 1
                    attacker_agent_score -= 1  # Penalize attacker for failing to insert a lasting bug
                    round_data["Defender Success"] = "Yes"
                    
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
                    round_data["Defender Success"] = "No"
                    
                    history = f"""
                    Attacker inserted a bug that changes output, and Defender failed to fix it.
                    Function Code:
                    {function_code}
                    Modified Code:
                    {modified_code_attacker}
                    Fixed Code:
                    {modified_code_defender}
                    """
                
                # Add history to DB
                history = str(history)
                document = Document(page_content=history)
                add_history(vector_store, document)
                round_data["History Added to DB"] = "Yes - Defender Ran"

        else:
            # Attacker failed initially
            history = f"""
            Attacker failed to insert a bug that changes output.
            Function Code:
            {function_code}
            Modified Code:
            {modified_code_attacker}
            """
            history = str(history)
            document = Document(page_content=history)
            add_history(vector_store, document)
            
            attacker_agent_score -= 1  # Penalize attacker for failing to insert a bug
            round_data["History Added to DB"] = "Yes - Attacker Failed"

        # Update scores in data
        round_data["Attacker Score (Total)"] = attacker_agent_score
        round_data["Defender Score (Total)"] = defender_agent_score
        
        # Append this round's data to the list
        results_data.append(round_data)

        print(f"Scores after Round {i+1}: Attacker: {attacker_agent_score}, Defender: {defender_agent_score}\n")

    # --- SAVE TO EXCEL ---
    print("Saving RAG Competition results to Excel...")
    df = pd.DataFrame(results_data)
    output_excel = "competition_results_nonAgentJudge+Rag.xlsx"
    df.to_excel(output_excel, index=False)
    print(f"Results saved to {output_excel}")

if __name__ == "__main__":
    asyncio.run(main())