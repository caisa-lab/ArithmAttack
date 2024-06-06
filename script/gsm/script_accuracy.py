from utils import calculate_accuracy

from config import access_token, DIR_PATH

output_files = [
       #(f"{DIR_PATH}/data/gsm/flan/flan_gsm_response.csv","flan_gsm"),
       (f"{DIR_PATH}/data/gsm/mistral/mistral_gsm_response.csv","mistral_gsm"),
       (f"{DIR_PATH}/data/gsm/mistral_instruct/mistral_instruct_gsm_response.csv","mistral_instruct"),
       (f"{DIR_PATH}/data/gsm/mistral_math/mistral_math_gsm_response.csv","mistral_math")

    ]

    # Call the calculate_accuracy function for each output file
for output_file in output_files:
	print(output_file)
	calculate_accuracy(output_file[0],output_file[1])