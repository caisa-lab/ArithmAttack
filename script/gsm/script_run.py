import subprocess
from config import DIR_PATH
import os


# List of scripts to run sequentially
scripts = [
    f"{DIR_PATH}/script/gsm/llama_gsm.py",
    f"{DIR_PATH}/script/gsm/mistral_instruct_gsm.py",
    f"{DIR_PATH}/script/gsm/mistral_math_gsm.py",
]

prompts = [
    "Always end the answer with 'The final answer is'",
]


# Function to create a command with prompts
def create_command(script, prompt):
    # Start with the python command and script name
    command = ["python", script, prompt]
    # Add each prompt as a new argument
    print(command)
    return command

# Make sure this is in sync with the script order
dir_name = ['llama','mistral_instruct','mistral_math']

# Run each script sequentially with the prompts
for script_pointer, script in enumerate(scripts):
    for i, prompt in enumerate(prompts):
        sc_name = os.path.splitext(os.path.basename(script))[0]
        print(sc_name)
        print("$$")
        print("Running:", script)
        # Create the command with the current script and all prompts
        cmd_line_args = f"{DIR_PATH}/data/multiArith/test_preprocessed.csv {DIR_PATH}/data/multiArith/{dir_name[script_pointer]}/{sc_name}_json_former_prompt_{i}.csv {prompt}"
        command = create_command(script,cmd_line_args)
        # Execute the command
        subprocess.run(command, shell=False)

print("All scripts have been executed.")
