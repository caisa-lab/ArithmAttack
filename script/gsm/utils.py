import pandas as pd
import re, sys

from config import access_token, DIR_PATH
import math

import tensorflow_hub as hub
import tensorflow as tf
# import tensorflow_text
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def get_questions_and_answer_from_dataset(csv_file_path):
    # Load the specific CSV file
    data = pd.read_csv(csv_file_path)

    # Extract the question column
    questions = data["question"].tolist()
    groundTruths = data["numeric_answer"].tolist()

    return questions, groundTruths

def get_questions_and_answer_from_multiArith_dataset(csv_file_path):
    # Load the specific CSV file
    data = pd.read_csv(csv_file_path)

    # Extract the question column
    questions = data["question"].tolist()
    groundTruths = data["answer"].tolist()

    return questions, groundTruths

def get_questions_and_answer_from_robustMath_dataset(json_file_path):
    # Load the specific JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Extract the questions and answers
    questions = [entry["x"] for entry in data]
    groundTruths = [entry["y"] for entry in data]

    return questions, groundTruths

def calculate_accuracy(output_file, name):

    print("Calculating accuracy")
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(output_file)

    # Calculate accuracy
    total_rows = len(df)


    def safe_convert_to_int(value):
        #print("value: ",value)
        if isinstance(value, (int, float)): 
            return float(value)
        try:
            #print("value: ",value)
            value = value.replace(',', '')
            
            # First try to find boxed numbers \boxed{number}
            boxed_match = re.search(r'\\boxed\{(\d+\.?\d*)\}', value)
            if boxed_match:
                return float(boxed_match.group(1))
            
            # Then try to find numbers in LaTeX math mode \(number\)
            latex_match = re.search(r'\\\(\s*(\d+\.?\d*)\s*\\\)', value)
            if latex_match:
                return float(latex_match.group(1))
            
            # Look for numbers after "Final Answer:"
            final_answer_match = re.search(r'Final Answer:.*?(\d+\.?\d*)', value, re.IGNORECASE)
            if final_answer_match:
                return float(final_answer_match.group(1))
            
            # If no specific format found, look for any number
            numbers = re.findall(r'\d+\.\d+|\d+', value)
            #print("numbers: ",numbers)
            if numbers:
                last_number = numbers[-1]
                return float(last_number)
            else:
                return sys.maxsize
        except (ValueError, TypeError):
            # Return a default value or handle the error as needed
            return None



    def safe_convert_llm_to_int(value):
        numeric_value = safe_convert_to_int(value)
        if numeric_value is not None and numeric_value > 192000000:
            return sys.maxsize
        return numeric_value

    # Type cast both columns to float
    print(f'###{name}##')
    df["Answer - Ground Truth Converted"] = df["Answer - Ground Truth"].apply(safe_convert_to_int)
    df["Answer - LLM Converted"] = df["Answer - LLM"].apply(safe_convert_llm_to_int)
    print(df["Answer - Ground Truth Converted"])
    print(df["Answer - LLM Converted"])

    correct_matches = sum(df["Answer - Ground Truth Converted"] == df["Answer - LLM Converted"])
    accuracy = correct_matches / total_rows * 100
    df.to_csv(
        f"{DIR_PATH}/data/multiArith/{name}/test_preprocessed.csv",
        index=False
    )



    print("correct_matches: ", correct_matches)
    print("accuracy: ", accuracy)
    # Create a DataFrame for script name and accuracy
    data = {"Script Name": [f"{name}.py noisy 30"], "Accuracy": [accuracy]}
    accuracy_df = pd.DataFrame(data)

    # Save the DataFrame to a new CSV file
    accuracy_df.to_csv(
        f"{DIR_PATH}/data/multiArith/accuracy.csv",
        mode="a",
        header=False,
        index=False,
    )

    print("Accuracy saved to accuracy.csv.")

# Function to calculate ASR
def calculate_asr(clean_df, attacked_df):
    total_correct = 0
    total_attacked_success = 0
    print('Length of clean and noisy df ',len(clean_df),len(attacked_df))
    for idx, row in tqdm(clean_df.iterrows(), total=len(clean_df), desc="Calculating ASR"):
        clean_answer = row['Answer - LLM Converted']
        ground_truth = row['Answer - Ground Truth Converted']
        attacked_answer = attacked_df.loc[idx, 'Answer - LLM Converted']
        
        # Check if clean answer was originally correct
        if clean_answer == ground_truth:
            total_correct += 1
            # Check if attack caused an incorrect answer
            if attacked_answer != ground_truth:
                total_attacked_success += 1
    print(total_correct,total_attacked_success)
    asr = total_attacked_success / total_correct if total_correct > 0 else 0
    return asr

# Function to calculate semantic similarity
def calculate_similarity(clean_df, attacked_df,use_model):
    total_similarity = 0
    num_questions = len(clean_df)
    
    # Use Universal Sentence Encoder to compute similarity
    clean_questions = clean_df['Question'].tolist()
    attacked_questions = attacked_df['Question'].tolist()
    
    # Get sentence embeddings
    clean_embeddings = use_model(clean_questions)
    attacked_embeddings = use_model(attacked_questions)
    
    for i in tqdm(range(num_questions), desc="Calculating Similarity"):
        if clean_questions[i] != attacked_questions[i]:
            print(clean_questions[i],attacked_questions[i])
        clean_vec = clean_embeddings[i].numpy()
        attacked_vec = attacked_embeddings[i].numpy()
        
        # Cosine similarity between original and attacked questions
        similarity = cosine_similarity([clean_vec], [attacked_vec])[0][0]
        # print('Similarity',similarity)
        total_similarity += similarity
    
    avg_similarity = total_similarity / num_questions if num_questions > 0 else 0
    return avg_similarity



def get_asr_and_similarity(clean_converted_file, noisy_converted_file,name):
    clean_converted = pd.read_csv(clean_converted_file)
    noisy_converted = pd.read_csv(noisy_converted_file)
    # Load the Universal Sentence Encoder
    embed = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/2")

    # Calculate ASR and Similarity
    asr = calculate_asr(clean_converted, noisy_converted)
    similarity = calculate_similarity(clean_converted, noisy_converted,embed)
    ASR = asr * 100.00
    print(f"Attack Success Rate (ASR): {ASR}%")
    print(f"Average Semantic Similarity: {similarity:.4f}")
    data = {"Script Name": [f"{name}.py noise 30"], "ASR": [ASR],"Similarity": [similarity]}
    accuracy_df = pd.DataFrame(data)

    # Save the DataFrame to a new CSV file
    accuracy_df.to_csv(
        f"{DIR_PATH}/data/RobustMath/asr_similarity.csv",
        mode="a",
        header=False,
        index=False,
    )

    print("ASR and Similarity saved to asr_similarity.csv.")