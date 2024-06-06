import pandas as pd
import json

# Path to the JSON file
json_file_path = "/Users/zainabedin/Desktop/nlplab/data/multiArith/questions.json"

# Read JSON data from the file
with open(json_file_path, 'r') as file:
    json_data = json.load(file)

# Extract 'sQuestion' and 'lSolutions' as 'question' and 'answer'
data = []
for item in json_data:
    question = item.get('sQuestion', '')
    answer = item.get('lSolutions', [])
    # Assuming there's only one solution in lSolutions, otherwise handle accordingly
    if answer:
        answer = answer[0]
    else:
        answer = None
    data.append({'question': question, 'answer': answer})

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
output_path = "/Users/zainabedin/Desktop/nlplab/data/multiArith/test_preprocessed.csv"
df.to_csv(output_path, index=False)

# Display the DataFrame
print(df.head())
