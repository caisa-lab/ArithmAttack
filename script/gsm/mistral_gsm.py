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
import math

from jsonformer import Jsonformer

from config import access_token, DIR_PATH
from utils import get_questions_and_answer_from_dataset

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

json_schema1 = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
    },
}

csv_file = f"{DIR_PATH}/data/gsm/test_preprocessed.csv"
questions, ground_truths = get_questions_and_answer_from_dataset(csv_file)


output_file = (
    f"{DIR_PATH}/data/gsm/mistral/mistral_gsm_response.csv"
)
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
        "answer": {"<contains the correct numerical answer>"},
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
        # qa_pipeline = pipeline(
        #     "text-generation",
        #     model=model,
        #     tokenizer=tokenizer,
        #     max_length=1024,
        #     do_sample=True,
        #     top_k=10,
        #     num_return_sequences=1,
        #     max_new_tokens = 1000
        # )
        # sequences = qa_pipeline(prompt, eos_token_id=tokenizer.eos_token_id)

        
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
