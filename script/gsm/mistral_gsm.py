import torch
import sys
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import csv
from config import access_token, DIR_PATH
from jsonformer import Jsonformer

from tqdm import tqdm

from utils import (
    get_questions_and_answer_from_multiArith_dataset,
    get_questions_and_answer_from_dataset)

json_schema1 = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
    },
}


access_token = access_token
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=access_token,
    device_map={"": 0},
    quantization_config=bnb_config,
)

model.config.use_cache = False
model.config.pretraining_tp = 1

args = sys.argv[1].split()

input_file = args[0]
output_file = args[1]
prompt = ' '.join(args[2:])

print('Input file ',input_file)
print('Output file ',output_file)
print('Prompt ',prompt)

#csv_file = f"{DIR_PATH}/data/multiArith/test_preprocessed.csv"
# csv_file = f"{DIR_PATH}data/gsm/sample_test_preprocessed.csv"


#uestions, ground_truths = get_questions_and_answer_from_multiArith_dataset(csv_file)
questions, ground_truths = get_questions_and_answer_from_dataset(input_file)

# output_file = (
#     f"{DIR_PATH}/data/multiArith/mistral/mistral_multiArith_response.csv"
# )
#output_file = f"{DIR_PATH}/data/gsm/mistral/mistral_gsm_response.csv"
# output_file = f"{DIR_PATH}/data/gsm/mistral/mistral_gsm_response_hugginface.csv"

# Command line arguments for prompts
# if len(sys.argv) > 1:
#     prompt = sys.argv[1:]  # Assume each argument is a separate prompt

counter = 0
with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = [
        "Question",
        "Answer - LLM",
        "Answer - Ground Truth",
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    counter = 0

    for question, ground_truth in tqdm(zip(questions, ground_truths), total=len(questions)):

        final_prompt = f"""
        [INST] 
        {prompt}

        question:
        {question}
        [/INST]
        """

        jsonformer = Jsonformer(
            model,
            tokenizer,
            json_schema1,
            final_prompt,
            max_number_tokens=1000,
            max_array_length=1000,
            max_string_token_length=1000,
            temperature = 0
        )
        generated_data = jsonformer()  
       
        import pprint
        pprint.pprint(final_prompt)
        print("##RESPONSE##")
        pprint.pprint(generated_data)
        print("\n\n")

        writer.writerow(
            {
                "Question": question,
                "Answer - Ground Truth": ground_truth,
                "Answer - LLM": generated_data,
            }
        )

        # counter += 1
        # if counter >= 5:
        #     break


print(f"Questions and answers saved to {output_file}")
