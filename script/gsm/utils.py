import pandas as pd
import re


def get_questions_and_answer_from_dataset(csv_file_path):
    # Load the specific CSV file
    data = pd.read_csv(csv_file_path)

    # Extract the question column
    questions = data["question"].tolist()
    groundTruths = data["numeric_answer"].tolist()

    return questions, groundTruths


def get_noisy_questions_and_answer_from_dataset(csv_file_path, aug=1):
    # Load the specific CSV file
    data = pd.read_csv(csv_file_path)

    # Extract the question column
    if aug == 1:
        questions = [eval(sublist)[0] for sublist in data['noisy_questions_aug_1']]
    else:
        questions = data[f"noisy_questions_aug_{aug}"].apply(eval).tolist()
    groundTruths = data["numeric_answer"].tolist()

    return questions, groundTruths

#Testing
#questions, _ = get_noisy_questions_and_answer_from_dataset('../../data/noisy_datasets/gsm8k_noisy_punct_10.csv',aug=4)
#print(questions)
