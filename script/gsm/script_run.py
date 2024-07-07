import subprocess
from config import DIR_PATH
import os


# List of scripts to run sequentially
scripts = [
    f"{DIR_PATH}/script/gsm/mistral_gsm.py",
    f"{DIR_PATH}/script/gsm/mistral_instruct_gsm.py",
    f"{DIR_PATH}/script/gsm/mistral_math_gsm.py",
    f"{DIR_PATH}/script/gsm/flan_gsm.py"
]

prompts = [
    "Always end the answer with 'The final answer is'",
    "Let's think step by step and always end the answer with 'The final answer is'.",
    "Always end the answer with 'The final answer is' and think step by step.",
    "Solve the following arithmetic problem step by step. Ensure to end the answer with 'The final answer is'.",
    "You are a math tutor. Solve the following arithmetic problem step by step. Always end the answer with 'The final answer is'.",
    "Think step by step through the following problem and clearly show each step of your reasoning. Ensure the final answer is clearly indicated by ending with 'The final answer is'.",
    "As a math tutor, explain your reasoning step by step for the following problem. Let's think step by step and end the answer with 'The final answer is'."
    "Answer the following questions. Give short answers."
]


# Function to create a command with prompts
def create_command(script, prompt):
    # Start with the python command and script name
    command = ["python", script, prompt]
    # Add each prompt as a new argument
    print(command)
    return command

# Make sure this is in sync with the script order
dir_name = ['mistral','mistral_instruct','mistral_math','flan']

# Run each script sequentially with the prompts
for script_pointer, script in enumerate(scripts):
    for i, prompt in enumerate(prompts):
        sc_name = os.path.splitext(os.path.basename(script))[0]
        print(sc_name)
        print("$$")
        print("Running:", script)
        # Create the command with the current script and all prompts
        cmd_line_args = f"{DIR_PATH}/data/gsm/sample_test_preprocessed.csv {DIR_PATH}/data/gsm/{dir_name[script_pointer]}/{sc_name}_json_former_prompt_{i}.csv {prompt}"
        command = create_command(script,cmd_line_args)
        # Execute the command
        subprocess.run(command, shell=False)

print("All scripts have been executed.")
