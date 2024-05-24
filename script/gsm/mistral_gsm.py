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

from script.gsm.config import access_token
from script.gsm.utils import get_questions_and_answer_from_dataset



DIR_PATH = "/home/stud/abedinz1/localDisk/nlplab"
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
        "answer": {"type":"string"},
    },
}

csv_file = f"{DIR_PATH}/data/train_preprocessed.csv"
questions, ground_truths = get_questions_and_answer_from_dataset(
    csv_file
)


output_file = f"/home/stud/abedinz1/localDisk/nlplab/data/gsm/mistral/mistral_gsm_response.csv"
counter=0
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
    for question, ground_truth in zip(
        questions, ground_truths
    ):

        prompt = f"""
        [INST] 
        persona:
        You are an expert in math problem solving
        
        goal:
        Please answer the following question:   

        Instruction:
        Make sure to give answer as numerical value only.

        question:
        {question}
        
        format:
        {json_format}


        [/INST]
        """
        
        jsonformer = Jsonformer(
            model,
            tokenizer,
            json_schema1,
            prompt,
            max_string_token_length=600,
        )

        generated_data = jsonformer()
        # import pprint
        # pprint.pprint(prompt)
        # print("##RESPONSE##")
        # pprint.pprint(generated_data)

        writer.writerow(
            {
                "Question": question,
                "Answer - Ground Truth": ground_truth,
                "Answer - LLM": generated_data["answer"],
            }
        )

        # counter += 1
        # if counter >=30:
        #     break


print(f"Questions and answers saved to {output_file}")

print("Calculating accuracy")
# Read the CSV file into a pandas DataFrame
df = pd.read_csv(output_file)

# Calculate accuracy
total_rows = len(df)

# Type cast both columns to float
df['Answer - Ground Truth'] = df['Answer - Ground Truth'].astype(float)
df['Answer - LLM'] = df['Answer - LLM'].astype(float)

print(df['Answer - Ground Truth'])
print(df['Answer - LLM'] )

correct_matches = sum(df['Answer - Ground Truth'] == df["Answer - LLM"])
accuracy = correct_matches / total_rows * 100


print("correct_matches: ", correct_matches)
print("accuracy: ",accuracy)
# Create a DataFrame for script name and accuracy
data = {'Script Name': ['lama3-8B-narrative-2.py'], 'Accuracy': [accuracy]}
accuracy_df = pd.DataFrame(data)

# Save the DataFrame to a new CSV file
#accuracy_df.to_csv('/home/stud/abedinz1/localDisk/narrative/data/gpqa/gpqa_results.csv', mode='a', header=False, index=False)

#print('Accuracy saved to accuracy.csv.')