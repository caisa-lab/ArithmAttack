import json
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import access_token
# Step 1: Load the local dataset
def load_local_dataset(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

access_token = access_token
dataset_path = "/Users/zainabedin/Desktop/NoiseInMath/data/train.json"  
data = load_local_dataset(dataset_path)

model_name = "mistralai/Mistral-7B-Instruct-v0.2"  
tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token)

def generate_answer(question):
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model.generate(**inputs)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

correct_answers = 0
total_questions = len(data)

for item in data:
    question = item['question']
    correct_answer = item['answer']
    
    generated_answer = generate_answer(question)
    
    if generated_answer.strip() == correct_answer.strip():
        correct_answers += 1

accuracy = correct_answers / total_questions * 100
print(f"Accuracy: {accuracy:.2f}%")