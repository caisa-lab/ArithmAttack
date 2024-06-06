import pandas as pd
import json

from config import DIR_PATH

# Path to the JSON file
json_file_path = f"{DIR_PATH}/data/multiArith/questions.json"

# Read JSON data from the file
with open(json_file_path, 'r') as file:
    json_data = json.load(file)

# Extract 'question' and 'final_ans' as 'question' and 'answer'
data = []
for item in json_data:
    question = item.get('question', '')
    answer = item.get('final_ans', [])
    data.append({'question': question, 'answer': answer})

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
output_path = f"{DIR_PATH}/data/multiArith/test_preprocessed.csv"
df.to_csv(output_path, index=False)

# Display the DataFrame
print(df.head())
