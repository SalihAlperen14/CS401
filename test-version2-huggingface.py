import sys
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# --- Check for bitsandbytes ---
try:
    import bitsandbytes
    print(f"Success: bitsandbytes version {bitsandbytes.__version__} found.")
except ImportError:
    print("Error: bitsandbytes not found.")
    sys.exit(1)

# --- 1. SETUP LOCAL MEMORY ---
print("\nInitializing Local Defender Memory...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma(
    collection_name="Defender_SWE_Memory",
    embedding_function=embeddings,
    persist_directory="./defender_memory_db_local"
)

def retrieve_experience(query: str, k: int = 2) -> str:
    docs = vector_store.similarity_search(query[:500], k=k)
    if not docs:
        return "No prior relevant experience found."
    return "\n\n".join([f"LESSON {i+1}: {d.page_content}" for i, d in enumerate(docs)])

def add_experience(lesson: str):
    vector_store.add_documents([Document(page_content=lesson)])

# --- 2. SETUP QWEN 14B MODEL ---
# Updated to the 14B Coder Model
MODEL_NAME = "Qwen/Qwen2.5-Coder-14B-Instruct" 

print(f"\nLoading {MODEL_NAME} in 4-bit mode...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True
)

# This "Streamer" will print tokens to the console as they are generated
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

def generate_agent_response(prompt: str, use_streamer=False) -> str:
    """Helper function to get a response from Qwen with optional streaming."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=1500,
        temperature=0.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    if use_streamer:
        generation_kwargs["streamer"] = streamer

    with torch.no_grad():
        outputs = model.generate(**generation_kwargs)
    
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

# --- 3. THE SWE-LITE LOOP ---
def main():
    print("\nLoading SWE-bench Verified...")
    dataset = load_dataset("SWE-bench/SWE-bench_Verified", split="test")
    
    for i in range(3):
        sample = dataset[i]
        issue_id = sample['instance_id']
        problem = sample['problem_statement']
        unit_tests = sample['test_patch']
        
        print(f"\n{'='*50}")
        print(f"🕵️ DEFENDER TACKLING ISSUE: {issue_id}")
        print(f"{'='*50}")
        
        # 1. Check Memory
        past_lessons = retrieve_experience(problem)
        print(f"🧠 Retrieved Memory: {past_lessons[:200]}...\n")
        
        # 2. Build Prompt
        prompt = f"""You are the Defender Agent, an expert software engineer.
FIX THE FOLLOWING ISSUE:
{problem}

TESTS TO PASS:
{unit_tests}

PAST LESSONS:
{past_lessons}

Provide your reasoning and the code fix in a python block.
"""
        
        # 3. Get the Defender's proposed fix (With Streaming!)
        print("🤖 Defender is thinking (Watch it type below):")
        print("-" * 30)
        defender_output = generate_agent_response(prompt, use_streamer=True)
        print("-" * 30)
        
        # 4. Memory Reflection Phase
        reflection_prompt = f"Summarize the bug and fix for {issue_id} in 3 short sentences for your future self memory. Do not use code."
        print("\n🧠 Updating memory...")
        lesson = generate_agent_response(reflection_prompt, use_streamer=False)
        print(f"Lesson saved: {lesson}")
        
        add_experience(f"Issue {issue_id}: {lesson}")

if __name__ == "__main__":
    main()