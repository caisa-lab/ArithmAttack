#from script.gsm.script_run import dir_name
from utils import calculate_accuracy,get_asr_and_similarity

from config import access_token, DIR_PATH

prompts = [
    #"Always end the answer with {The final answer is}",
    #"Let's think step by step and always end the answer with {The final answer is}.",
    #"Always end the answer with {The final answer is} and think step by step.",
    #"Solve the following arithmetic problem step by step. Ensure to end the answer with {The final answer is}.",
    #"You are a math tutor. Solve the following arithmetic problem step by step. Always end the answer with {The final answer is}.",
    "Think step by step through the following problem and clearly show each step of your reasoning. Ensure the final answer is clearly indicated by ending with {The final answer is}.",
    #"As a math tutor, explain your reasoning step by step for the following problem. Let's think step by step and end the answer with {The final answer is}.",
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


#dir_names = ['mistral_instruct', 'mistral_math', 'llama']
# for dir_name in dir_names:
#     for i, _ in enumerate(prompts):
#         response_file_path = f"{DIR_PATH}/data/gsm/{dir_name}/{dir_name}_gsm_noisy_punct_10_prompt_{i}.csv"
#         print("response_file_path: ",response_file_path)
#         print("dir_name: ",dir_name)
#         calculate_accuracy(response_file_path, dir_name)


all_dir_names = ['DeepSeek-R1-Distill-Llama-8B']

# for dir_name in all_dir_names:
#     response_file_path = f"{DIR_PATH}/data/multiArith/{dir_name}/{dir_name}.csv"
#     print("response_file_path: ",response_file_path)
#     print("dir_name: ",dir_name)
#     calculate_accuracy(response_file_path, dir_name)

for dir_name in all_dir_names:
    for percent in [10,30,50]:
        clean_file_path_converted = f"{DIR_PATH}/data/multiArith/{dir_name}/{dir_name}.csv"
        attacked_file_path_converted = f"{DIR_PATH}/data/multiArith/{dir_name}/{dir_name}_noisy_punct_{percent}.csv"
        print("clean_file_path_converted: ",clean_file_path_converted)
        print("attacked_file_path_converted: ",attacked_file_path_converted)
        print("dir_name: ",dir_name)
        #calculate_accuracy(attacked_file_path_converted, dir_name,percent)
        get_asr_and_similarity(clean_file_path_converted, attacked_file_path_converted,dir_name,percent)
    
# dir_names = ['Llama-3.1-8B-Instruct','gemma-2-2b-it','gemma-2-2b-jpn-it','zephyr-7b-beta','Qwen2.5-1.5B-Instruct']

# for dir_name in dir_names:
#     for i, _ in enumerate(prompts):
#         response_file_path = f"{DIR_PATH}/data/gsm/{dir_name}/{dir_name}_gsm_noisy_punct_10_prompt_{i}.csv"
#         print("response_file_path: ",response_file_path)
#         print("dir_name: ",dir_name)
#         calculate_accuracy(response_file_path, dir_name)

# dir_names = ['Llama-3.1-8B-Instruct','gemma-2-2b-it','gemma-2-2b-jpn-it','zephyr-7b-beta','Qwen2.5-1.5B-Instruct']
# prev_dir_names = ['llama','mistral_instruct','mistral_math']

# dir = ['Mistral-7B-Instruct-v0.2']
# for dir_name in dir:
#     for percent in [10,30,50]:
#         response_file_path = f"{DIR_PATH}/data/multiArith/{dir_name}/{dir_name}_noisy_punct_{percent}.csv"
#         print("response_file_path: ",response_file_path)
#         print("dir_name: ",dir_name)
#         calculate_accuracy(response_file_path, dir_name,percent)


#all_dir_names = ['llama','mistral_math','Mistral-7B-Instruct-v0.2','Llama-3.1-8B-Instruct','gemma-2-2b-it','zephyr-7b-beta','Qwen2.5-1.5B-Instruct']

# for dir_name in all_dir_names:
#     for percent in [10,30,50]:
#         clean_file_path_converted = f"{DIR_PATH}/data/multiArith/{dir_name}/{dir_name}_converted.csv"
#         attacked_file_path_converted = f"{DIR_PATH}/data/multiArith/{dir_name}/{dir_name}_noisy_{percent}_converted.csv"
#         print("clean_file_path_converted: ",clean_file_path_converted)
#         print("attacked_file_path_converted: ",attacked_file_path_converted)
#         print("dir_name: ",dir_name)
#         get_asr_and_similarity(clean_file_path_converted, attacked_file_path_converted,dir_name,percent)