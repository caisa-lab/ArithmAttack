import subprocess

# List of scripts to run sequentially
scripts = [
    "mistral_gsm.py",
    "mistral_instruct_gsm.py",
    "mistral_math_gsm.py",
    "flan_gsm.py"
]

prompts = [
    "Let's think step by step and always end the answer with {The final answer is}.",
    "What is the capital of France?",
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
