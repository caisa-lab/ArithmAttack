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

from tqdm import tqdm
from jsonformer import Jsonformer

from config import access_token, DIR_PATH
from utils import get_noisy_questions_and_answer_from_multi_arith_dataset


access_token = access_token
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
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

json_schema1 = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
    },
}

csv_file = f"{DIR_PATH}/data/noisy_datasets/multiArith_test_noisy_punct_10.csv"
questions, ground_truths = get_noisy_questions_and_answer_from_multi_arith_dataset(csv_file)




output_file = f"{DIR_PATH}/data/multiArith/mistral_instruct/mistral_instruct_multiArith_response.csv"

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
    for question, ground_truth in tqdm(zip(questions, ground_truths), total=len(questions)):

        # prompt = f"""
        # [INST] 
        # persona:
        # You are an expert in math problem solving
        
        # goal:
        # Please answer the following question:   

        # Instruction:
        # Make sure to give answer as numerical value only.

        # question:
        # {question}
        
        # format:
        # {json_format}


        # [/INST]
        # """
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

        #pprint.pprint(prompt)
        #print("##RESPONSE##")
        #pprint.pprint(generated_data)

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
