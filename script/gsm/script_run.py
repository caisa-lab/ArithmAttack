import subprocess
from config import DIR_PATH
import os


# List of scripts to run sequentially
scripts = [
    # f"{DIR_PATH}/script/gsm/mistral_gsm.py",
    f"{DIR_PATH}/script/gsm/llama_gsm.py",
    f"{DIR_PATH}/script/gsm/mistral_math_gsm.py",
    f"{DIR_PATH}/script/gsm/mistral_instruct_gsm.py",
    # f"{DIR_PATH}/script/gsm/flan_gsm.py"
]

prompts = [
    # "Always end the answer with {The final answer is}",
    # "Let's think step by step and always end the answer with {The final answer is}.",
    # "Always end the answer with {The final answer is} and think step by step.",
    # "Solve the following arithmetic problem step by step. Ensure to end the answer with {The final answer is}.",
    # "You are a math tutor. Solve the following arithmetic problem step by step. Always end the answer with {The final answer is}.",
    "Think step by step through the following problem and clearly show each step of your reasoning. Ensure the final answer is clearly indicated by ending with {The final answer is}.",
    # "As a math tutor, explain your reasoning step by step for the following problem. Let's think step by step and end the answer with {The final answer is}."
]

prompt = "Think step by step through the following problem and clearly show each step of your reasoning. Ensure the final answer is clearly indicated by ending with {The final answer is}."


model_names = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-2b-it",
    "google/gemma-2-2b-jpn-it",
    "HuggingFaceH4/zephyr-7b-beta",
    "Qwen/Qwen2.5-1.5B-Instruct"
]


# Function to create a command with prompts
def create_command(script, prompt):
    # Start with the python command and script name
    command = ["python", script, prompt]
    # Add each prompt as a new argument
    print(command)
    return command


# Make sure this is in sync with the script order
# dir_name = ['llama','mistral_math','mistral_instruct',]


# # Run each script sequentially with the prompts
# for script_pointer, script in enumerate(scripts):
#     for i, prompt in enumerate(prompts):
#         sc_name = os.path.splitext(os.path.basename(script))[0]
#         print(sc_name)
#         print("$$")
#         print("Running:", script)
#         # Create the command with the current script and all prompts
#         cmd_line_args = f"{DIR_PATH}/data/gsm/test_preprocessed.csv {DIR_PATH}/data/gsm/{dir_name[script_pointer]}/{sc_name}_prompt_{i}.csv {prompt}"
#         command = create_command(script,cmd_line_args)
#         # Execute the command
#         subprocess.run(command, shell=False)

# print("All scripts have been executed.")


for model in model_names:
    model_name = model.split("/")[1]
    cmd_line_args = f"{DIR_PATH}/data/multiArith/test_preprocessed.csv {DIR_PATH}/data/multiArith/{model_name}/{model_name}.csv {model} {prompt}"
    script = "generic_model_script.py"
    command = create_command(script, cmd_line_args)
    # Execute the command
    subprocess.run(command, shell=False)
