import torch
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import csv

from config import access_token, DIR_PATH

from tqdm import tqdm

from utils import (
    get_questions_and_answer_from_multiArith_dataset,
    get_questions_and_answer_from_dataset,
    get_questions_and_answer_from_noisy_dataset,
    get_questions_and_answer_from_robustMath_dataset
)


args = sys.argv[1].split()

input_file = args[0]
output_file = args[1]
model_name = args[2]
prompt = " ".join(args[3:])

print("Model Name: ", model_name)
print("Input file ", input_file)
print("Output file ", output_file)
print("Prompt ", prompt)

# Check CUDA availability
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This script requires a GPU.")

logger.info(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")

access_token = access_token
tokenizer = AutoTokenizer.from_pretrained(
    model_name, token=access_token, trust_remote_code=True
)

try:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    logger.info("Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=access_token,
        device_map={"": 0},
        quantization_config=bnb_config,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    
    # Verify model is on GPU
    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("Model is not on GPU despite device_map configuration")
    
    logger.info(f"Model loaded successfully on device: {next(model.parameters()).device}")
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# csv_file = f"{DIR_PATH}/data/multiArith/test_preprocessed.csv"
# csv_file = f"{DIR_PATH}data/gsm/sample_test_preprocessed.csv"

# questions, ground_truths = get_questions_and_answer_from_multiArith_dataset(input_file)
# questions, ground_truths = get_questions_and_answer_from_dataset(input_file)
#questions, ground_truths = get_questions_and_answer_from_noisy_dataset(input_file)
questions, ground_truths = get_questions_and_answer_from_robustMath_dataset(input_file)

# output_file = f"{DIR_PATH}/data/multiArith/mistral_instruct/mistral_instruct_multiArith_response.csv"
# output_file = f"{DIR_PATH}/data/gsm/mistral_instruct/mistral_instruct_gsm_response.csv"
# output_file = f"{DIR_PATH}/data/gsm/mistral_instruct/mistral_instruct_gsm_response_hugginface.csv"

# Command line arguments for prompts
# if len(sys.argv) > 1:
#     prompt = sys.argv[1:]  # Assume each argument is a separate prompt

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

    for question, ground_truth in tqdm(
        zip(questions, ground_truths), total=len(questions)
    ):
        messages = [
            {
                "role": "user",
                "content": f"""
                    {prompt}
                    question:
                    {question}
                """,
            }
        ]
        try:
            model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
            generated_ids = model.generate(
                model_inputs, max_new_tokens=1000, do_sample=True
            )
            generated_data = tokenizer.batch_decode(generated_ids)[0]
        except Exception as e:
            logger.error(f"Error generating response for question {counter}: {str(e)}")
            generated_data = "ERROR: Failed to generate response"

        # import pprint
        # print("##MESSAGE##")
        # pprint.pprint(messages)
        # print("##RESPONSE##")
        # pprint.pprint(generated_data)

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
