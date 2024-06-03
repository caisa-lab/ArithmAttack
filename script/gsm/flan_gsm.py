
from transformers import T5Tokenizer, T5ForConditionalGeneration

import csv
import zipfile
import pandas as pd
import re
import random
import sys, os, json

from jsonformer import Jsonformer

from config import access_token, DIR_PATH
from utils import get_questions_and_answer_from_dataset

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained(
    "google/flan-t5-xl", device_map={"": 0}, max_length=512
)


json_schema1 = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
    },
}

csv_file = f"{DIR_PATH}/data/gsm/test_preprocessed.csv"
questions, ground_truths = get_questions_and_answer_from_dataset(csv_file)


output_file = (
    f"{DIR_PATH}/data/gsm/flan/flan_gsm_response.csv"
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
        # if counter >= 1:
        #     break


print(f"Questions and answers saved to {output_file}")
