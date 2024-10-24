import pandas as pd
import re, sys
import json
from config import access_token, DIR_PATH
import math


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

def calculate_accuracy(output_file, name,percent):

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
        f"{DIR_PATH}/data/multiArith/{name}/{name}_converted_1.csv",
        index=False
    )



    print("correct_matches: ", correct_matches)
    print("accuracy: ", accuracy)
    # Create a DataFrame for script name and accuracy
    data = {"Script Name": [f"{name}.py"], "Accuracy": [accuracy]}
    accuracy_df = pd.DataFrame(data)

    # Save the DataFrame to a new CSV file
    accuracy_df.to_csv(
        f"{DIR_PATH}/data/multiArith/accuracy.csv",
        mode="a",
        header=False,
        index=False,
    )

    print("Accuracy saved to accuracy.csv.")
