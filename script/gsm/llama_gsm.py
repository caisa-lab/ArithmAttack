import torch
import sys

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import csv

from config import access_token, DIR_PATH

from tqdm import tqdm

from utils import (
    get_questions_and_answer_from_multiArith_dataset,
    get_questions_and_answer_from_dataset)

access_token = access_token
model_name = "meta-llama/Meta-Llama-3-8B"
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

print('Input file ', input_file)
print('Output file ', output_file)
print('Prompt ', prompt)

# csv_file = f"{DIR_PATH}/data/multiArith/test_preprocessed.csv"
# csv_file = f"{DIR_PATH}data/gsm/sample_test_preprocessed.csv"

questions, ground_truths = get_questions_and_answer_from_multiArith_dataset(input_file)
# questions, ground_truths = get_questions_and_answer_from_dataset(input_file)


# output_file = f"{DIR_PATH}/data/multiArith/mistral_instruct/mistral_instruct_multiArith_response.csv"
# output_file = f"{DIR_PATH}/data/gsm/mistral_instruct/mistral_instruct_gsm_response.csv"
# output_file = f"{DIR_PATH}/data/gsm/mistral_instruct/mistral_instruct_gsm_response_hugginface.csv"

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
        messages = [
            {
                "role": "user",
                "content": f"""
                    {prompt}
                    question:
                    {question}
                """
            }
        ]

        model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
        generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
        generated_data = tokenizer.batch_decode(generated_ids)[0]

        # import pprint
        # print("##MESSAGE##")
        # pprint.pprint(messages)
        # print("##RESPONSE##")
        # pprint.pprint(generated_data)

        writer.writerow(
            {
                "Question": question,
                "Answer - Ground Truth": ground_truth,
                "Answer - LLM": generated_data,
            }
        )

        # counter += 1
        # if counter >= 2:
        #     break

print(f"Questions and answers saved to {output_file}")
