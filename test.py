import difflib
import re
import ast
from llama_cpp import Llama

# --- DOSYA İSİMLERİ ---
BUGGY_FILE = "kth-b.py"
GOLD_FILE = "kth.py"
MODEL_PATH = r"C:\Users\hp\AppData\Local\nomic.ai\GPT4All\Meta-Llama-3-8B-Instruct.Q4_0.gguf"

# --- MODEL YÜKLE ---
print("🤖 Model yükleniyor...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=8192,
    n_gpu_layers=-1,
    verbose=False,
    n_threads=8
)

# -------------------------------
#  FONKSİYON: Koddan parametre sayısı bul
# -------------------------------
def get_param_count(code_content):
    try:
        tree = ast.parse(code_content)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                return len(node.args.args)
    except:
        return None


# -------------------------------
#  FONKSİYON: Koddan fonksiyon adı bul
# -------------------------------
def get_function_name(code_content):
    try:
        tree = ast.parse(code_content)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                return node.name
    except:
        return None


# -------------------------------
#  FONKSİYON: Modelden fix edilmiş kodu al
# -------------------------------
def fix_code(buggy_content):
    prompt = f"""<|start_header_id|>system<|end_header_id|>
Fix bugs in this Python code. Return ONLY the corrected Python code.
<|eot_id|><|start_header_id|>user<|end_header_id|>
{buggy_content}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    output = llm(prompt, max_tokens=400, temperature=0.1)
    raw = output["choices"][0]["text"]

    # Markdown temizleme
    raw = raw.replace("```python", "").replace("```", "").strip()

    # Son fonksiyon bloğunu al
    blocks = list(re.finditer(
        r"def\s+\w+\s*\(.*?\):[\s\S]*?(?=^def|\Z)",
        raw,
        re.MULTILINE
    ))

    fixed = blocks[-1].group().strip() if blocks else raw.strip()

    print("\n🔧 Model tarafından düzeltilen kod:")
    print(fixed)
    print()
    return fixed


# -------------------------------
#  TEST INPUT ÜRETME (OTOMATİK PARAMETRE ALGILAMA)
# -------------------------------
def generate_inputs(gold_content):
    param_count = get_param_count(gold_content)

    if param_count == 1:
        system_prompt = """
Generate exactly 5 test inputs for this function.
Return ONLY a Python list of 5 values.
No explanation.
"""
    else:
        system_prompt = f"""
Generate exactly 5 test inputs for the function.
Each test input must be a tuple with {param_count} arguments.
Return ONLY a Python list of tuples.
No explanation.
"""

    prompt = f"""<|start_header_id|>system<|end_header_id|>
{system_prompt}
<|eot_id|><|start_header_id|>user<|end_header_id|>
{gold_content}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    output = llm(prompt, max_tokens=200, temperature=0.1)
    text = output["choices"][0]["text"].strip()
    return text


# -------------------------------
#  RUN & COMPARE – Altın & Düzeltilmiş kodu karşılaştır
# -------------------------------
def run_and_compare(fixed_code, gold_code, input_list_str):
    try:
        test_inputs = ast.literal_eval(input_list_str)
    except:
        return 0, f"Input listesi parse edilemedi: {input_list_str}"

    func_name = get_function_name(gold_code)
    if not func_name:
        return 0, "Fonksiyon adı bulunamadı."

    param_count = get_param_count(gold_code)

    env_gold = {}
    env_fixed = {}

    try:
        exec(gold_code, env_gold)
    except Exception as e:
        return 0, f"Gold kod hata verdi: {str(e)}"

    try:
        exec(fixed_code, env_fixed)
    except Exception as e:
        return 0, f"Model kodu hata verdi: {str(e)}"

    func_gold = env_gold.get(func_name)
    func_fixed = env_fixed.get(func_name)

    print(f"\n🔍 Fonksiyon: {func_name}")
    print(f"📊 Test Girdileri: {test_inputs}")
    print("-" * 40)

    passed = 0
    total = len(test_inputs)

    for inp in test_inputs:
        try:
            # Tek parametreli fonksiyon
            if param_count == 1:
                res_gold = func_gold(inp)
                res_fixed = func_fixed(inp)

            # Çok parametreli fonksiyon
            else:
                if not isinstance(inp, tuple):
                    return 0, f"Input tuple olmalıydı, ama şu geldi: {inp}"
                res_gold = func_gold(*inp)
                res_fixed = func_fixed(*inp)

        except Exception as e:
            res_gold = f"HATA: {e}"
            res_fixed = f"HATA: {e}"

        if res_gold == res_fixed:
            print(f"✅ {inp} -> {res_gold}")
            passed += 1
        else:
            print(f"❌ {inp} -> GOLD: {res_gold} | MODEL: {res_fixed}")

    success = (passed / total) * 100
    return success, f"{passed}/{total} test geçti."


# -------------------------------
#  ANA ÇALIŞMA
# -------------------------------
try:
    with open(BUGGY_FILE, "r", encoding="utf-8") as f:
        buggy_txt = f.read()
    with open(GOLD_FILE, "r", encoding="utf-8") as f:
        gold_txt = f.read()

    print("\n1️⃣  Model kodu düzeltiyor...")
    fixed_code = fix_code(buggy_txt)

    print("\n2️⃣  Test girdileri üretiliyor...")
    inputs = generate_inputs(gold_txt)
    print(f"Üretilen girdiler: {inputs}")

    print("\n3️⃣  Kodlar karşılaştırılıyor...")
    score, msg = run_and_compare(fixed_code, gold_txt, inputs)

    print("\n" + "=" * 30)
    print(f"BAŞARI SKORU: %{score:.2f}")
    print("=" * 30)
    print(msg)

    if score == 100:
        print("🏆 MÜKEMMEL!")
    elif score > 0:
        print("⚠️ Kısmi başarı")
    else:
        print("💥 HATA")

except FileNotFoundError:
    print("❌ Dosyalar bulunamadı.")
