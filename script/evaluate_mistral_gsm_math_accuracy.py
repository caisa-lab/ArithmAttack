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


from config import access_token



DIR_PATH = "/home/stud/abedinz1/localDisk/narrative"
access_token = access_token
#model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
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

# Create the question-answering pipeline
qa_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=1024,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
)

prompt = "What is the capital of india"

sequences = qa_pipeline(prompt, eos_token_id=tokenizer.eos_token_id)
print(sequences)

