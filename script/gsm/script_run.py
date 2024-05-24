import subprocess

# List of scripts to run sequentially
scripts = [
    'python  /home/stud/abedinz1/localDisk/nlplab/script/gsm/mistral_gsm.py',
    'python  /home/stud/abedinz1/localDisk/nlplab/script/gsm/mistral_instruct_gsm.py',
    'python  /home/stud/abedinz1/localDisk/nlplab/script/gsm/mistral_math_gsm.py',
]

# Run each script sequentially
for script in scripts:
    print("$$")
    print(script)
    subprocess.run(script, shell=True)

print("All scripts have been executed.")
