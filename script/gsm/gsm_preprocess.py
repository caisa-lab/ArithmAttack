import pandas as pd
import re
import json


# Initialize an empty list to hold the data
data = []

# Open the JSONL file and read line by line
with open("/home/stud/abedinz1/localDisk/nlplab/data/gsm/test.jsonl", 'r') as file:
    for line in file:
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            continue


# Load the dataset
df = pd.DataFrame(data)

def extract_numeric_answer(answer):
    # Use a regex pattern that matches an optional sign (+ or -) before the digits
    numeric_answer = re.search(r"#### ([+-]?\d+)", answer)
    if numeric_answer:
        return int(numeric_answer.group(1))
    else:
        return None

df["numeric_answer"] = df["answer"].apply(extract_numeric_answer)
df.to_csv("/home/stud/abedinz1/localDisk/nlplab/data/gsm/test_preprocessed.csv", index=False)

# Display the preprocessed dataset
print(df.head())
