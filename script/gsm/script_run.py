import subprocess

# List of scripts to run sequentially
scripts = [
    "python  mistral_gsm.py",
    "python  mistral_instruct_gsm.py",
    "python  mistral_math_gsm.py",
    "python  flan_gsm.py"
]

# Run each script sequentially
for script in scripts:
    print("$$")
    print(script)
    subprocess.run(script, shell=True)

print("All scripts have been executed.")
