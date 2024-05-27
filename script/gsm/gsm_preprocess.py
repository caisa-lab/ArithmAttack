import pandas as pd
import re

# Load the dataset
df = pd.read_json("/home/stud/abedinz1/localDisk/nlplab/data/gsm/train.json")

def extract_numeric_answer(answer):
    # Use a regex pattern that matches an optional sign (+ or -) before the digits
    numeric_answer = re.search(r"#### ([+-]?\d+)", answer)
    if numeric_answer:
        return int(numeric_answer.group(1))
    else:
        return None

df["numeric_answer"] = df["answer"].apply(extract_numeric_answer)
df.to_csv("/home/stud/abedinz1/localDisk/nlplab/data/gsm/train_preprocessed.csv", index=False)

# Display the preprocessed dataset
print(df.head())
