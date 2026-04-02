import os
import subprocess
from datasets import load_dataset

# 1. Load the dataset
print("Loading dataset...")
sbf = load_dataset('SWE-bench/SWE-bench_verified')
sample = sbf['train'][0]  # Let's just look at the first issue as an example

repo_name = sample['repo']          # e.g., 'scikit-learn/scikit-learn'
base_commit = sample['base_commit'] # The commit hash WITH the bug
patch = sample['patch']             # The code diff that FIXES the bug

print(f"Target Repo: {repo_name}")
print(f"Buggy Commit: {base_commit}")

# 2. Clone the repository from GitHub
repo_url = f"https://github.com/{repo_name}.git"
repo_dir = repo_name.split('/')[-1] # Gets the folder name, e.g., 'scikit-learn'

if not os.path.exists(repo_dir):
    print(f"Cloning {repo_url}...")
    subprocess.run(["git", "clone", repo_url, repo_dir], check=True)

# 3. Checkout the specific commit that has the bug
print("Checking out the buggy commit...")
subprocess.run(["git", "checkout", base_commit], cwd=repo_dir, check=True)

# --- AT THIS POINT, THE REPO HAS THE BUG ---

# 4. Save the fix patch to a file
patch_filename = os.path.abspath("fix.patch")
with open(patch_filename, "w") as f:
    f.write(patch)

# 5. Apply the patch to fix the bug
print("Applying the patch to get the bug-free code...")
try:
    subprocess.run(["git", "apply", patch_filename], cwd=repo_dir, check=True)
    print("Success! The repository is now bug-free.")
except subprocess.CalledProcessError:
    print("Failed to apply the patch. You might need to use 'git am' or check line endings.")

# Clean up the patch file if you want
# os.remove(patch_filename)