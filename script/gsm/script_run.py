import subprocess
from config import DIR_PATH

# List of scripts to run sequentially
scripts = [
    f"{DIR_PATH}/script/gsm/mistral_gsm.py",
    f"{DIR_PATH}/script/gsm/mistral_instruct_gsm.py",
    f"{DIR_PATH}/script/gsm/mistral_math_gsm.py",
    f"{DIR_PATH}/script/gsm/flan_gsm.py"
]

prompts = [
    "Let's think step by step and always end the answer with {The final answer is}.",
    " Always end the answer with {The final answer is} and think step by step.",
]

# Function to create a command with prompts
def create_command(script, prompt):
    # Start with the python command and script name
    command = ["python", script, prompt]
    # Add each prompt as a new argument
    print(command)
    return command

# Run each script sequentially with the prompts
for script in scripts:
    for prompt in prompts:
        print("$$")
        print("Running:", script)
        # Create the command with the current script and all prompts
        command = create_command(script, prompt)
        # Execute the command
        subprocess.run(command, shell=False)

print("All scripts have been executed.")
