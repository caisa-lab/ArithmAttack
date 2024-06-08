import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import csv
import zipfile
import pandas as pd
import re
import random
import sys, os, json

from jsonformer import Jsonformer

from config import access_token, DIR_PATH
from utils import (
    get_questions_and_answer_from_multiArith_dataset,
    get_questions_and_answer_from_dataset)


access_token = access_token
model_name = "meta-math/MetaMath-Mistral-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": 0},
    quantization_config=bnb_config,
)

model.config.use_cache = False
model.config.pretraining_tp = 1

json_schema1 = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
    },
}

#csv_file = f"{DIR_PATH}/data/multiArith/test_preprocessed.csv"
csv_file = f"{DIR_PATH}/data/gsm/test_preprocessed.csv"

#questions, ground_truths = get_questions_and_answer_from_multiArith_dataset(csv_file)
questions, ground_truths = get_questions_and_answer_from_dataset(csv_file)


#output_file = f"{DIR_PATH}/data/multiArith/mistral_math/mistral_math_multiArith_response.csv"
output_file = f"{DIR_PATH}/data/gsm/mistral_math/mistral_math_gsm_response.csv"


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

    json_format = {
        "answer": {},
    }
    for question, ground_truth in zip(questions, ground_truths):

        prompt = f"""
        [INST] 
        Let's think step by step and always end the answer with 'The final answer is'.

        question:
        {question}
        [/INST]
        """
        jsonformer = Jsonformer(
            model,
            tokenizer,
            json_schema1,
            prompt,
            max_number_tokens=1000,
            max_array_length=1000,
            max_string_token_length=1000,
            temperature = 0
        )

        generated_data = jsonformer()
        
        import pprint
        pprint.pprint(prompt)
        print("##RESPONSE##")
        pprint.pprint(generated_data)

        writer.writerow(
            {
                "Question": question,
                "Answer - Ground Truth": ground_truth,
                "Answer - LLM": generated_data["answer"],
            }
        )
        # counter += 1
        # if counter >= 4:
        #     break

print(f"Questions and answers saved to {output_file}")