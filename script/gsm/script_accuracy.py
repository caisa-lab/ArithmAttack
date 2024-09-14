from utils import calculate_accuracy

from config import access_token, DIR_PATH

prompts = [
    "Always end the answer with {The final answer is}",
    "Let's think step by step and always end the answer with {The final answer is}.",
    "Always end the answer with {The final answer is} and think step by step.",
    "Solve the following arithmetic problem step by step. Ensure to end the answer with {The final answer is}.",
    "You are a math tutor. Solve the following arithmetic problem step by step. Always end the answer with {The final answer is}.",
    "Think step by step through the following problem and clearly show each step of your reasoning. Ensure the final answer is clearly indicated by ending with {The final answer is}.",
    "As a math tutor, explain your reasoning step by step for the following problem. Let's think step by step and end the answer with {The final answer is}.",
]

# response_files = [
#        (f"{DIR_PATH}/data/gsm/flan/flan_gsm_response.csv","flan_gsm"),
#        (f"{DIR_PATH}/data/gsm/mistral/mistral_gsm_response.csv","mistral_gsm"),
#        (f"{DIR_PATH}/data/gsm/mistral_instruct/mistral_instruct_gsm_response.csv","mistral_instruct"),
#        (f"{DIR_PATH}/data/gsm/mistral_math/mistral_math_gsm_response.csv","mistral_math")

#     ]

# output_files = [
#        (f"{DIR_PATH}/data/multiArith/mistral_math/mistral_math_gsm_response.csv","mistral_math"),
#        (f"{DIR_PATH}/data/multiArith/mistral/mistral_multiArith_response.csv","mistral_gsm"),
#        (f"{DIR_PATH}/data/multiArith/mistral_instruct/mistral_instruct_gsm_response.csv","mistral_instruct"),
#        (f"{DIR_PATH}/data/multiArith/flan/flan_gsm_response.csv","flan_gsm"),
#     ]

    # Call the calculate_accuracy function for each output file


dir_names = ['mistral','mistral_instruct','mistral_math','llama','flan']
#dir_names = ['mistral']

for dir_name in dir_names:
    for i, _ in enumerate(prompts):
        response_file_path = f"{DIR_PATH}/data/gsm/{dir_name}/{dir_name}_gsm_json_former_prompt_{i}.csv"
        print("response_file_path: ",response_file_path)
        print("dir_name: ",dir_name)
        calculate_accuracy(response_file_path, dir_name)