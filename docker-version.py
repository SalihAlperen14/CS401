import re
import subprocess
import time
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# Alpine + apk: avoids Debian apt "Hash Sum mismatch" inside Docker on some networks (proxy/AV/CDN).
# The host runs Python; the container only needs shell tools for the agent.
_DOCKER_BASE_IMAGE = "alpine:3.20"


def _apk_install_tools(container_name: str, attempts: int = 3) -> None:
    """Install git/bash/grep/sed via Alpine apk (more reliable than apt when mirrors misbehave)."""
    script = "apk update && apk add --no-cache git bash grep sed"
    for i in range(attempts):
        res = subprocess.run(
            ["docker", "exec", container_name, "sh", "-c", script],
            capture_output=True,
            text=True,
        )
        if res.returncode == 0:
            return
        if i < attempts - 1:
            print(f"⚠️ apk failed (attempt {i + 1}/{attempts}); retrying...")
            time.sleep(2)
    print(res.stderr or res.stdout or "(no output)")
    res.check_returncode()


# --- 1. DOCKER ENVIRONMENT (The Sandbox ACI) ---
class DockerEnvironment:
    def __init__(self, container_name="swe_sandbox"):
        self.name = container_name
        self.workdir = "/app/repo"
        print(f"📦 Initializing Docker Sandbox: {self.name}...")

        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)

        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                self.name,
                _DOCKER_BASE_IMAGE,
                "sleep",
                "infinity",
            ],
            check=True,
        )

        print("🔧 Installing required tools in Docker (this takes a moment)...")
        _apk_install_tools(self.name)

    def setup_repository(self, repo_url: str, base_commit: str):
        """Clones the repo and checks out the specific historical commit safely."""
        print(f"📥 Downloading {repo_url} (this may take a minute for large repos)...")

        subprocess.run(["docker", "exec", self.name, "mkdir", "-p", "/app"], check=True)

        subprocess.run(
            [
                "docker",
                "exec",
                self.name,
                "git",
                "clone",
                f"https://github.com/{repo_url}.git",
                self.workdir,
            ],
            check=True,
        )

        print(f"🕰️ Reverting code to historical commit {base_commit}...")
        subprocess.run(
            ["docker", "exec", "-w", self.workdir, self.name, "git", "checkout", base_commit],
            check=True,
        )
        print("✅ Repository successfully prepared!")

    def execute(self, command: str) -> str:
        """Executes a bash command acting as the AI's hands (cwd = repo)."""
        res = subprocess.run(
            ["docker", "exec", "-w", self.workdir, self.name, "bash", "-c", command],
            capture_output=True,
            text=True,
        )

        output = res.stdout + res.stderr

        max_length = 2000
        if len(output) > max_length:
            output = output[:max_length] + "\n\n... [OUTPUT TRUNCATED BY ACI] ..."

        if not output.strip():
            output = "[Command executed successfully with no output]"

        return output

    def close(self):
        """Cleans up the container."""
        print("🧹 Cleaning up Docker sandbox...")
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)


# --- 2. MEMORY & BRAIN SETUP ---
print("\n🧠 Initializing Brain and Memory...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(
    collection_name="Defender_SWE_Memory",
    embedding_function=embeddings,
    persist_directory="./defender_memory_db_local",
)

llm = OllamaLLM(model="qwen2.5-coder:14b", temperature=0.2)


def retrieve_experience(query: str, k: int = 2) -> str:
    docs = vector_store.similarity_search(query[:500], k=k)
    return "\n\n".join([f"LESSON: {d.page_content}" for d in docs]) if docs else "No prior experience."


def add_experience(lesson: str):
    vector_store.add_documents([Document(page_content=lesson)])


# --- 3. THE SWE-AGENT LOOP ---
def main():
    print("Loading SWE-bench Verified...")
    dataset = load_dataset("SWE-bench/SWE-bench_Verified", split="test")

    sample = dataset[0]
    issue_id = sample["instance_id"]
    problem = sample["problem_statement"]
    repo_name = sample["repo"]
    base_commit = sample["base_commit"]

    print(f"\n{'=' * 50}\n🕵️ DEFENDER TACKLING ISSUE: {issue_id}\n{'=' * 50}")

    env = DockerEnvironment()
    env.setup_repository(repo_name, base_commit)

    past_lessons = retrieve_experience(problem)

    system_prompt = f"""You are an autonomous AI Software Engineer.
Your task is to fix the following GitHub issue:
{problem}

Past lessons learned:
{past_lessons}

ENVIRONMENT RULES:
You have a bash terminal. The repository is in your current directory.
You must explore the code, find the bug, and edit the files.
To run a command, output it in a bash block like this:
```bash
grep -rn "def suspect_function" .
```
Only output ONE bash block per turn.
When you are completely finished fixing the bug, output the exact string: COMPLETE_TASK_AND_SUBMIT
"""

    history = f"System:\n{system_prompt}\n"
    max_steps = 10

    try:
        for step in range(max_steps):
            print(f"\n--- Turn {step + 1}/{max_steps} ---")
            print("🤖 Defender is thinking...")

            agent_response = llm.invoke(history + "\nAgent:")
            print(f"Agent Output:\n{agent_response}")
            history += f"\nAgent:\n{agent_response}\n"

            if "COMPLETE_TASK_AND_SUBMIT" in agent_response:
                print("\n✅ Agent claims the bug is fixed!")
                break

            bash_match = re.search(r"```bash\n(.*?)```", agent_response, re.DOTALL)
            if bash_match:
                command = bash_match.group(1).strip()
                print(f"💻 Executing in Docker: {command}")

                observation = env.execute(command)
                print(f"🔍 Observation:\n{observation}")

                history += f"Environment Observation:\n{observation}\n"
            else:
                print("⚠️ No bash block found. Prompting agent to correct format.")
                history += (
                    "Environment Observation:\nError: You did not provide a bash block. "
                    "Please provide a ```bash ... ``` block.\n"
                )

    finally:
        print("\n🧠 Updating memory...")
        reflection_prompt = (
            f"Based on this trajectory, summarize how to fix issue {issue_id} in 2 sentences. "
            "Do not include code."
        )
        lesson = llm.invoke(history + "\nSystem:\n" + reflection_prompt)
        add_experience(f"Issue {issue_id}: {lesson}")
        print(f"Lesson saved: {lesson}")

        env.close()


if __name__ == "__main__":
    main()
