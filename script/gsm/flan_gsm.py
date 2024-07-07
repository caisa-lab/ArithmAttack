
from transformers import T5Tokenizer, T5ForConditionalGeneration
import csv
import sys

from config import access_token, DIR_PATH
from jsonformer import Jsonformer

from tqdm import tqdm


from utils import (
    get_questions_and_answer_from_multiArith_dataset,
    get_questions_and_answer_from_dataset)

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained(
    "google/flan-t5-xl", device_map={"": 0}, max_length=512
)
json_schema1 = {
    "type": "object",
    "properties": {
        "answer": {"type": "number"},
    },
}

args = sys.argv[1].split()

input_file = args[0]
output_file = args[1]
prompt = ' '.join(args[2:])

print('Input file ',input_file)
print('Output file ',output_file)
print('Prompt ',prompt)


# # Command line arguments for prompts
# if len(sys.argv) > 1:
#     prompt = sys.argv[1:]  # Assume each argument is a separate prompt


#csv_file = f"{DIR_PATH}/data/multiArith/test_preprocessed.csv"
# csv_file = f"{DIR_PATH}data/gsm/sample_test_preprocessed.csv"

#questions, ground_truths = get_questions_and_answer_from_multiArith_dataset(csv_file)
questions, ground_truths = get_questions_and_answer_from_dataset(input_file)

# output_file = (
#     f"{DIR_PATH}/data/multiArith/flan/flan_multiArith_response.csv"
# )
# output_file = (
#     f"{DIR_PATH}/data/gsm/flan/flan_gsm_response.csv"
# )
# output_file = (
#     f"{DIR_PATH}/data/gsm/flan/flan_gsm_response_hugginface.csv"
# )

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

    for question, ground_truth in tqdm(zip(questions, ground_truths), total=len(questions)):

        final_prompt = f"""
        [INST] 
        {prompt}

        ### question ###
        {question}

        ### answer ###
        [/INST]
        """

        jsonformer = Jsonformer(
            model,
            tokenizer,
            json_schema1,
            final_prompt,
            max_number_tokens=5000,
            max_array_length=5000,
            max_string_token_length=5000,
            temperature = 0
        )
        generated_data = jsonformer()  
       
        import pprint
        pprint.pprint(final_prompt)
        print("##RESPONSE##")
        pprint.pprint(generated_data)
        print("\n\n")

        writer.writerow(
            {
                "Question": question,
                "Answer - Ground Truth": ground_truth,
                "Answer - LLM": generated_data,
            }
        )

        # counter += 1
        # if counter >= 1:
        #     break


print(f"Questions and answers saved to {output_file}")
