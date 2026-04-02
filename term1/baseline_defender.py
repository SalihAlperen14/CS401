import os
from dotenv import load_dotenv
import re
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
import subprocess
from typing import Dict, Optional
import pandas as pd

load_dotenv()
my_api_key = os.getenv("OPENAI_API_KEY")
if not my_api_key:
    raise RuntimeError("OPENAI_API_KEY not found in environment. Put it in .env or export it.")

from openai import OpenAI

client = OpenAI(api_key=my_api_key)
docker_executor = DockerCommandLineCodeExecutor(work_dir="coding")
docker_tool = PythonCodeExecutionTool(docker_executor)
folder_path = "buggy_python_programs"


def docker_runner(script_path: str,
                  image: str = "python:3.11-slim",
                  workdir_in_container: str = "/app",
                  timeout: Optional[int] = None) -> Dict[str, object]:
    """Run a Python script in a Docker container."""
    host_script = os.path.abspath(script_path)
    if not os.path.isfile(host_script):
        raise FileNotFoundError(f"Script not found: {host_script}")

    host_dir = os.path.dirname(host_script)
    script_basename = os.path.basename(host_script)

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{host_dir}:{workdir_in_container}",
        "-w", workdir_in_container,
        image,
        "python", script_basename
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    return {
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "returncode": proc.returncode
    }


def list_function_files(folder_path="Functions"):
    """List all Python files in the folder."""
    return [f for f in os.listdir(folder_path) if f.endswith(".py")]


def read_function_file(filepath):
    """Read file content."""
    with open(filepath, "r") as f:
        return f.read().strip()


def extract_code_blocks(response_text):
    """
    Extract fixed code and test case from LLM response with robust parsing.
    Returns (fixed_code, test_case, error_message)
    """
    # Try to find Fixed Code section
    fixed_match = re.search(r"Fixed Code:\s*```python\n(.*?)```", response_text, re.DOTALL)
    
    # Try to find Test Case section
    test_match = re.search(r"Test Case:\s*```python\n(.*?)```", response_text, re.DOTALL)
    
    # Alternative: look for any code blocks if structured format fails
    if not fixed_match or not test_match:
        all_code_blocks = re.findall(r"```python\n(.*?)```", response_text, re.DOTALL)
        
        if len(all_code_blocks) >= 2:
            fixed_code = all_code_blocks[0].strip()
            test_case = all_code_blocks[1].strip()
            return fixed_code, test_case, None
        elif len(all_code_blocks) == 1:
            return None, None, "Only one code block found, expected two (fixed code and test case)"
        else:
            return None, None, "No code blocks found in response"
    
    fixed_code = fixed_match.group(1).strip() if fixed_match else None
    test_case = test_match.group(1).strip() if test_match else None
    
    if not fixed_code:
        return None, None, "Fixed code not found"
    if not test_case:
        return None, None, "Test case not found"
    
    return fixed_code, test_case, None


def clean_output(output_str):
    """Clean output string for comparison."""
    if output_str is None:
        return ""
    return output_str.strip()


# Main execution
function_files = list_function_files(folder_path)
number_of_rounds = 5

for q in range(number_of_rounds):
    results_data = []

    for i, function_file in enumerate(function_files):
        round_data = {
            "Round": i + 1,
            "File Name": function_file,
            "Defender Output": "",
            "Expected Output (Defender Phase)": "",
            "Defender Success": "No",
            "Error": ""
        }
        
        print(f"\n{'='*60}")
        print(f"Round {q+1}, File {i+1}/{len(function_files)}: {function_file}")
        print('='*60)
        
        # File paths
        main_code_file = os.path.join("coding", "main_code.py")
        modified_file_defender = os.path.join("coding", "defender_modified.py")
        test_runner_file_defender = os.path.join("coding", "test_runner_defender.py")
        test_runner_file_judge = os.path.join("coding", "test_runner_judge.py")
        
        # Read buggy and correct function code
        function_code = read_function_file(os.path.join(folder_path, function_file))
        
        try:
            function_code_correct = read_function_file(os.path.join("Functions", function_file))
        except FileNotFoundError:
            print(f"Warning: Correct version not found for {function_file}")
            round_data["Error"] = "Correct version not found"
            results_data.append(round_data)
            continue
        
        # Get defender response
        try:
            defender_full_response = client.responses.create(
                model="gpt-4o",
                input=f"""Can you find the bug in the following code and fix it?:
{function_code}

After you have fixed the bug, generate a single test case that prints output to verify the fix.

Return STRICTLY in the following format with NO additional text:
Fixed Code:
```python
<Fixed code here>
```
Test Case:
```python
<test case here>
```
"""
            )
            defender_full_response_text = defender_full_response.output_text
            print(f"\n--- Defender Response ---")
            print(defender_full_response_text[:500] + "..." if len(defender_full_response_text) > 500 else defender_full_response_text)
            
        except Exception as e:
            print(f"Error getting defender response: {e}")
            round_data["Error"] = f"API Error: {str(e)}"
            results_data.append(round_data)
            continue
        
        # Extract code blocks
        modified_code_defender, test_case_defender, parse_error = extract_code_blocks(defender_full_response_text)
        
        if parse_error:
            print(f"Parse error: {parse_error}")
            round_data["Error"] = parse_error
            results_data.append(round_data)
            continue
        
        # Write files
        try:
            with open(modified_file_defender, "w", encoding="utf-8") as f:
                f.write(modified_code_defender.rstrip() + "\n")
            
            with open(main_code_file, "w", encoding="utf-8") as f:
                f.write(function_code_correct.rstrip() + "\n")
            
            with open(test_runner_file_defender, "w", encoding="utf-8") as f:
                f.write("from defender_modified import *\n\n")
                f.write(test_case_defender + "\n")
            
            with open(test_runner_file_judge, "w", encoding="utf-8") as f:
                f.write("from main_code import *\n\n")
                f.write(test_case_defender + "\n")
        except Exception as e:
            print(f"Error writing files: {e}")
            round_data["Error"] = f"File write error: {str(e)}"
            results_data.append(round_data)
            continue
        
        # Run tests
        try:
            res_defender = docker_runner("coding/test_runner_defender.py", timeout=10)
        except subprocess.TimeoutExpired:
            print("Defender's test case execution timed out.")
            res_defender = {"stdout": "", "stderr": "Timeout", "returncode": -1}
        except Exception as e:
            print(f"Error running defender test: {e}")
            res_defender = {"stdout": "", "stderr": str(e), "returncode": -1}
        
        try:
            res_judge = docker_runner("coding/test_runner_judge.py", timeout=10)
        except subprocess.TimeoutExpired:
            print("Judge's test case execution timed out.")
            res_judge = {"stdout": "", "stderr": "Timeout", "returncode": -1}
        except Exception as e:
            print(f"Error running judge test: {e}")
            res_judge = {"stdout": "", "stderr": str(e), "returncode": -1}
        
        # Process outputs
        defender_output = clean_output(res_defender["stdout"]) or clean_output(res_defender["stderr"])
        expected_output = clean_output(res_judge["stdout"])
        
        round_data["Defender Output"] = defender_output
        round_data["Expected Output (Defender Phase)"] = expected_output
        
        print(f"\n--- Test Results ---")
        print(f"Defender Output:\n{defender_output}")
        print(f"\nExpected Output:\n{expected_output}")
        
        # Check success
        if defender_output == expected_output and expected_output != "":
            round_data["Defender Success"] = "Yes"
            print("\n✓ SUCCESS: Outputs match!")
        else:
            print("\n✗ FAILURE: Outputs don't match")
        
        results_data.append(round_data)
    
    # Save results
    df = pd.DataFrame(results_data)
    output_excel = f"defender_baseline_results_round_{q+1}.xlsx"
    df.to_excel(output_excel, index=False)
    print(f"\n{'='*60}")
    print(f"Round {q+1} complete. Results saved to {output_excel}")
    print(f"Success rate: {df['Defender Success'].value_counts().get('Yes', 0)}/{len(df)}")
    print('='*60)