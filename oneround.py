# agent_competition.py
import asyncio
import os
from dotenv import load_dotenv

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent, CodeExecutorAgent
from autogen_agentchat.teams import SelectorGroupChat, Swarm
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.ui import Console
from typing import Any, Dict, List
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.tools.code_execution import PythonCodeExecutionTool

from autogen_core import DefaultTopicId, MessageContext, RoutedAgent, default_subscription, message_handler
from autogen_core.code_executor import CodeBlock, CodeExecutor
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    LLMMessage,
    SystemMessage,
    UserMessage,
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

docker_tool = PythonCodeExecutionTool(DockerCommandLineCodeExecutor(work_dir="coding"))
# Agents
attacker_agent = AssistantAgent(
    name="AttackerAgent",
    description="Attack the code to insert bug.",
    model_client=model_client,
    system_message=(
        """
        You are responsible for inserting a bug into the code you are given. 
        You are competing with defender agent which detects the bugs. Insert subtle, 
        non-syntax bugs so the code still runs but produces incorrect behavior. 
        Don't reveal where the bug is — just return the modified code.
        Save the code under coding directory named attacker.py
        At the end, send changed code to JudgeAgent.
        """
    ),
)

defender_agent = AssistantAgent(
    name="DefenderAgent",
    description="Fix the code in case there is a bug.",
    model_client=model_client,
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
    reflect_on_tool_use = True ,
    system_message="""
        You are a judge that makes AttackerAgent and DefenderAgent compete.
        Initial Score: AttackerAgent=0, DefenderAgent=0.
        Original Code: (The code provided in the task).
        Test Case: **You must make up a test case** (e.g., input arguments) for the function. Store this test case and the **expected output of the Original Code** using your code execution tool.
        
        **Flow and Scoring Rules:**

        1.  **Initial Action:** When you receive the original code, **immediately ask AttackerAgent to insert a bug and return only the modified code.**
        2.  **After AttackerAgent sends modified code:**
            * **EXECUTE:** Use your code execution tool to run your Test Case on the *Modified Code*.
            * **EVALUATE:** Compare the output of the Modified Code with the expected output of the Original Code.
            * **If the outputs are DIFFERENT (Attacker Success):** State the new score, then send the **Modified Code** to DefenderAgent and ask them to fix it.
            * **If the outputs are the SAME (Attacker Fail):** AttackerAgent failed. Subtract 1 point from AttackerAgent. State the new score. Write **TERMINATE** to end this round.

        3.  **After DefenderAgent sends fixed code:**
            * **EXECUTE:** Use your code execution tool to run your Test Case on the *Fixed Code*.
            * **EVALUATE:** Compare the output of the Fixed Code with the expected output of the Original Code.
            * **If the outputs are the SAME (Defender Success):** Add 1 point to DefenderAgent and subtract 1 point from AttackerAgent. State the new score. Write **TERMINATE** to end this round.
            * **If the outputs are DIFFERENT (Defender Fail):** Add 1 point to AttackerAgent and subtract 1 point from DefenderAgent. State the new score. Write **TERMINATE** to end this round.
        
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


task = """
Can you make agents compete for the following code?:
def find_in_sorted(arr, x):
    def binsearch(start, end):
        if start == end:
            return -1
        mid = start + (end - start) // 2
        if x < arr[mid]:
            return binsearch(start, mid)
        elif x > arr[mid]:
            return binsearch(mid + 1, end)
        else:
            return mid

    return binsearch(0, len(arr))
"""

async def main() -> None:
    async with docker_tool._executor:
        await Console(team.run_stream(task=task))
if __name__ == "__main__":
    asyncio.run(main())
