import pandas as pd
import re

def get_questions_and_answer_from_dataset(csv_file_path):
    # Load the specific CSV file
    data = pd.read_csv(csv_file_path)

    # Extract the question column
    questions = data["question"].tolist()
    groundTruths = data["numeric_answer"].tolist()


    return questions, groundTruths