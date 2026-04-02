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
from term1.modifiedAgents import LastMessageDefenderAgent
from autogen_ext.memory.chromadb import (
    ChromaDBVectorMemory,
    PersistentChromaDBVectorMemoryConfig,
    SentenceTransformerEmbeddingFunctionConfig,
)
import random

chroma_user_memory = ChromaDBVectorMemory(
        config=PersistentChromaDBVectorMemoryConfig(
            collection_name="preferences",
            k=2,  # Return top k results
            score_threshold=0.4,  # Minimum similarity score
            embedding_function_config=SentenceTransformerEmbeddingFunctionConfig(
                model_name="all-MiniLM-L6-v2"  # Use default model for testing
            ),
        )
    )

load_dotenv()
my_api_key = os.getenv("OPENAI_API_KEY")
if not my_api_key:
    raise RuntimeError("OPENAI_API_KEY not found in environment. Put it in .env or export it.")

# model client used by all agents
model_client = OpenAIChatCompletionClient(
    model="gpt-4o-2024-08-06",
    api_key=my_api_key,
)

docker_executor = DockerCommandLineCodeExecutor(work_dir="coding")
docker_tool = PythonCodeExecutionTool(docker_executor)
# Agents
attacker_agent = AssistantAgent(
    name="AttackerAgent",
    description="Attack the code to insert bug.",
    model_client=model_client,
    memory=[chroma_user_memory],
    system_message=(
        """
        You are responsible for inserting a bug into the code you are given. 
        You are competing with defender agent which detects the bugs. Insert subtle, 
        non-syntax bugs so the code still runs but produces incorrect behavior. 
        Don't reveal where the bug is — just return the modified code.
        Don't add any explanations or comments even in python script, just return the modified code.
        At the end, send changed code to JudgeAgent.
        """
    ),
)

defender_agent = LastMessageDefenderAgent(
    name="DefenderAgent",
    description="Fix the code in case there is a bug.",
    model_client=model_client,
    memory=[chroma_user_memory],
    system_message=(
        """
        You are responsible for fixing the given code if there are bugs.
        At the end, send changed code to JudgeAgent.
        """
    ),
)

# Assuming you choose the safer Docker execution setup

judge_agent = AssistantAgent(
    name="JudgeAgent",
    description="Controls the competition flow, runs test cases using its code execution tool, and keeps the score.",
    model_client=model_client,
    tools=[docker_tool], 
    reflect_on_tool_use = True,
    memory=[chroma_user_memory],
    system_message="""
        You are a judge that makes AttackerAgent and DefenderAgent compete.
        Initial Score: AttackerAgent=0, DefenderAgent=0. You can retrieve the most recent score from the most recent round.
        Original Code: (The code provided in the task).
        Test Case: **You must make up a test case** (e.g., input arguments) for the function. Store this test case and the **expected output of the Original Code** using your code execution tool.
        
        **Flow and Scoring Rules:**

        1.  **Initial Action:** When you receive the original code, **immediately ask AttackerAgent to insert a bug and return only the modified code.**
        2.  **After AttackerAgent sends modified code:**
            * **EXECUTE:** Use your code execution tool to run your Test Case on the *Modified Code*.
            * **EVALUATE:** Compare the output of the Modified Code with the expected output of the Original Code.
            * **If the outputs are DIFFERENT (Attacker Success):** State the new score, then send the **Modified Code** to DefenderAgent and ask them to fix it. When you ask DefenderAgent to fix the code, make sure to add **Modified Code**.
            * **If the outputs are the SAME (Attacker Fail):** AttackerAgent failed. Subtract 1 point from AttackerAgent. State the new score, pure code, attacked code. Write **TERMINATE** to end this round.

        3.  **After DefenderAgent sends fixed code:**
            * **EXECUTE:** Use your code execution tool to run your Test Case on the *Fixed Code*.
            * **EVALUATE:** Compare the output of the Fixed Code with the expected output of the Original Code.
            * **If the outputs are the SAME (Defender Success):** Add 1 point to DefenderAgent and subtract 1 point from AttackerAgent. State the new score, whole pure code, whole attacked code and whole defended code. Write **TERMINATE** to end this round.
            * **If the outputs are DIFFERENT (Defender Fail):** Add 1 point to AttackerAgent and subtract 1 point from DefenderAgent. State the new score, whole pure code, whole attacked code and whole defended code. Write **TERMINATE** to end this round.
        
        **Always explicitly address the next agent in your turn.**
        """
)

# Termination conditions
text_mention_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=10)
termination = text_mention_termination | max_messages_termination

selector_prompt = """Select an agent to perform the next step of the competition.
The JudgeAgent must use its code execution tool to check the behavior of the code provided by the Attacker or Defender.

{roles}

Current conversation context:
{history}

Read the above conversation, then select an agent from {participants} to perform the next task.
Only select one agent.
"""

team = SelectorGroupChat(
    [attacker_agent, defender_agent, judge_agent],
    model_client=model_client,
    termination_condition=termination,
    selector_prompt=selector_prompt,
    allow_repeated_speaker=True,
)




# --- Load all available function files ---
def list_function_files(folder_path="Functions"):
    return [f for f in os.listdir(folder_path) if f.endswith(".py")]

def read_function_file(filepath):
    with open(filepath, "r") as f:
        return f.read().strip()


async def main() -> None:
    folder_path = "functions"
    function_files = list_function_files(folder_path)

    if not function_files:
        print("No function files found under 'functions/'")
        return

    # 🔁 For each round, pick one function to compete on
    for i in range(5):
        print(f"--- Round {i+1} ---")

        # Pick a random function file (or sequentially if you prefer)
        func_file = random.choice(function_files)
        func_path = os.path.join(folder_path, func_file)
        function_code = read_function_file(func_path)

        # Create the round-specific task
        task = f"""
Can you make agents compete for the following code?:
{function_code}
"""

        async with docker_tool._executor:
            a = await Console(team.run_stream(task=task))

        last_message_content = str(a.messages[-1].content)
        filtered_content = last_message_content.replace("TERMINATE", "").strip()
        filtered_content = "ROUND " + str(i+1) + " RESULT:\n" + filtered_content
        await chroma_user_memory.add(
            MemoryContent(
                content=filtered_content,
                mime_type=MemoryMimeType.TEXT
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
