import asyncio
import json
import os
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from gpt4all import GPT4All

# -------------------
# Model paths
# -------------------
MODEL_FOLDER = r"C:\Users\Huawei\Desktop\Letta Projects\.venv\models\Llama-3.2-3B-Instruct"
MODEL_FILE = "Llama-3.2-3B-Instruct-Q4_0.gguf"

STATE_DIR = "states"
os.makedirs(STATE_DIR, exist_ok=True)

# -------------------
# Utility functions for JSON persistence
# -------------------
def save_json(filename, data):
    """Save dictionary as JSON to disk."""
    path = os.path.join(STATE_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f)

def load_json(filename):
    """Load dictionary from JSON if file exists."""
    path = os.path.join(STATE_DIR, filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

# -------------------
# GPT4All client wrapper
# -------------------
class GPT4AllClient:
    def __init__(self, folder_path, model_file):
        self.model = GPT4All(
            model_name=model_file,
            model_path=folder_path,
            allow_download=False
        )

    async def generate(self, prompt: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.model.generate(prompt))

# Initialize once
injector_client = GPT4AllClient(MODEL_FOLDER, MODEL_FILE)

# -------------------
# Async state helpers
# -------------------
async def save_agent_state(agent, filename):
    state = await agent.save_state()
    save_json(filename, state)

async def load_agent_state(agent, filename):
    state = load_json(filename)
    if state:
        await agent.load_state(state)
        print(f"✅ Loaded state from {filename}")

# -------------------
# Create the injector agent
# -------------------
async def create_injector():
    agent = AssistantAgent(
        name="FaultInjector",
        system_message="You are a code fault injector. Inject subtle bugs in the code.",
        model_client=injector_client
    )
    await load_agent_state(agent, "injector_state.json")
    return agent

# -------------------
# Run one conversation round
# -------------------
async def run_round(code: str, round_num: int):
    injector = await create_injector()

    team = RoundRobinGroupChat(
        [injector],
        termination_condition=MaxMessageTermination(max_messages=1)
    )

    # Load previous team state (if exists)
    team_state = load_json("team_state.json")
    if team_state:
        await team.load_state(team_state)
        print("✅ Loaded previous team state")

    print(f"\n--- ROUND {round_num} ---")
    print("Original Code:\n", code)

    # Run the chat stream
    async for msg in team.run_stream(task=f"Inject subtle bugs into this code:\n{code}"):
        if hasattr(msg, "content"):
            print("\nInjected Bug:\n", msg.content)

    # Save agent + team state
    await save_agent_state(injector, "injector_state.json")
    team_state = await team.save_state()
    save_json("team_state.json", team_state)
    print("💾 States saved to disk")

# -------------------
# Main
# -------------------
async def main():
    snippets = [
        "def add(a, b): return a + b",
        "def multiply(a, b): return a * b",
        "def divide(a, b): return a / b"
    ]
    for i, code in enumerate(snippets, start=1):
        await run_round(code, i)

if __name__ == "__main__":
    asyncio.run(main())
