
from transformers import T5Tokenizer, T5ForConditionalGeneration
import csv
import sys

from config import access_token, DIR_PATH
from utils import (
    get_questions_and_answer_from_multiArith_dataset,
    get_questions_and_answer_from_dataset)

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained(
    "google/flan-t5-xl", device_map={"": 0}, max_length=512
)


#csv_file = f"{DIR_PATH}/data/multiArith/test_preprocessed.csv"
csv_file = f"{DIR_PATH}/data/gsm/test_preprocessed.csv"

#questions, ground_truths = get_questions_and_answer_from_multiArith_dataset(csv_file)
questions, ground_truths = get_questions_and_answer_from_dataset(csv_file)

# output_file = (
#     f"{DIR_PATH}/data/multiArith/flan/flan_multiArith_response.csv"
# )
# output_file = (
#     f"{DIR_PATH}/data/gsm/flan/flan_gsm_response.csv"
# )
output_file = (
    f"{DIR_PATH}/data/gsm/flan/flan_gsm_response_hugginface.csv"
)

# Command line arguments for prompts
if len(sys.argv) > 1:
    prompt = sys.argv[1:]  # Assume each argument is a separate prompt

counter = 0
with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = [
        "Question",
        "Answer - LLM",
        "Answer - Ground Truth",
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    counter = 0

    for question, ground_truth in zip(questions, ground_truths):

        messages = [
            {
                "role": "user", 
                "content": f"""
                    {prompt}
                    question:
                    {question}
                """
            }
        ]

        model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
        generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
        generated_data=tokenizer.batch_decode(generated_ids)[0]
        
        import pprint
        print("##MESSAGE##")
        pprint.pprint(messages)
        print("##RESPONSE##")
        pprint.pprint(generated_data)

        writer.writerow(
            {
                "Question": question,
                "Answer - Ground Truth": ground_truth,
                "Answer - LLM": generated_data,
            }
        )

        counter += 1
        if counter >= 4:
            break


print(f"Questions and answers saved to {output_file}")
