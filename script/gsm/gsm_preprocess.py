import pandas as pd
import re

# Load the dataset
df = pd.read_json("../../data/train.json")


def extract_numeric_answer(answer):
    numeric_answer = re.search(r"#### (\d+)", answer)
    if numeric_answer:
        return int(numeric_answer.group(1))
    else:
        return None


df["numeric_answer"] = df["answer"].apply(extract_numeric_answer)
df.to_csv("../data/train_preprocessed.csv", index=False)
# Display the preprocessed dataset
print(df.head())
