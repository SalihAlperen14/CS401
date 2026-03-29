import sys
import os
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM # New: Using Ollama for the brain

# --- 1. SETUP LOCAL MEMORY (Stays the same, very fast) ---
print("\nInitializing Local Defender Memory...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma(
    collection_name="Defender_SWE_Memory",
    embedding_function=embeddings,
    persist_directory="./defender_memory_db_local"
)

def retrieve_experience(query: str, k: int = 2) -> str:
    docs = vector_store.similarity_search(query[:500], k=k)
    return "\n\n".join([f"LESSON {i+1}: {d.page_content}" for i, d in enumerate(docs)]) if docs else "No prior experience."

# --- 2. SETUP OLLAMA BRAIN (Instant Load) ---
print("Connecting to Ollama (Qwen2.5-Coder-14B)...")
# This doesn't load the shards; it just connects to the Ollama service already running
llm = OllamaLLM(model="qwen2.5-coder:14b", temperature=0.2)

# --- 3. THE SWE-LITE LOOP ---
def main():
    print("Loading SWE-bench Verified...")
    dataset = load_dataset("SWE-bench/SWE-bench_Verified", split="test")
    
    for i in range(3):
        sample = dataset[i]
        issue_id = sample['instance_id']
        problem = sample['problem_statement']
        unit_tests = sample['test_patch']
        
        print(f"\n{'='*50}\n🕵️ DEFENDER TACKLING ISSUE: {issue_id}\n{'='*50}")
        
        # 1. Memory
        past_lessons = retrieve_experience(problem)
        
        # 2. Prompt
        prompt = f"System: You are an expert engineer.\nIssue: {problem}\nTests: {unit_tests}\nPast Lessons: {past_lessons}\nFix the bug and provide code in a python block."
        
        # 3. Think (This will be much faster now!)
        print("🤖 Defender is thinking...")
        defender_output = llm.invoke(prompt) 
        print(f"\n--- DEFENDER OUTPUT ---\n{defender_output}\n")
        
        # 4. Reflection & Memory Update
        reflection_prompt = f"Summarize the fix for {issue_id} in 3 short sentences for memory."
        lesson = llm.invoke(reflection_prompt)
        vector_store.add_documents([Document(page_content=f"Issue {issue_id}: {lesson}")])
        print(f"✅ Memory updated with lesson.")

if __name__ == "__main__":
    main()