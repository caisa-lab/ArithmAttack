import subprocess
from config import DIR_PATH

# List of scripts to run sequentially
scripts = [
    f"python  {DIR_PATH}/script/gsm/mistral_gsm.py",
    f"python  {DIR_PATH}/script/gsm/mistral_instruct_gsm.py",
    f"python  {DIR_PATH}/script/gsm/mistral_math_gsm.py",
    f"python  {DIR_PATH}/script/gsm/flan_gsm.py"
]

# Run each script sequentially
for script in scripts:
    print("$$")
    print(script)
    subprocess.run(script, shell=True)

print("All scripts have been executed.")
