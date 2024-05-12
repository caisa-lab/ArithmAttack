import pandas as pd
import re

# Load the dataset
df = pd.read_json("/Users/zainabedin/Desktop/NoiseInMath/data/train.jsonl", lines=True)

# Define a function to extract numeric answer from the "answer" field
def extract_numeric_answer(answer):
    numeric_answer = re.search(r'#### (\d+)', answer)
    if numeric_answer:
        return int(numeric_answer.group(1))
    else:
        return None

# Apply the function to create a new column "numeric_answer"
df['numeric_answer'] = df['answer'].apply(extract_numeric_answer)

# Display the preprocessed dataset
print(df.head())

# Save the preprocessed dataset if needed
# df.to_json("preprocessed_train.jsonl", orient='records', lines=True)
