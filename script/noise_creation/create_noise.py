# AEDA: An Easier Data Augmentation Technique for Text classification
# Akbar Karimi, Leonardo Rossi, Andrea Prati

import random
from tqdm import tqdm
import pandas as pd

from config import DIR_PATH

tqdm.pandas()

random.seed(0)



PUNCTUATIONS = ['.', ',', '!', '?', ';', ':']
#DATASETS = ['cr', 'sst2', 'subj', 'pc', 'trec']
DATASET = '../../data/gsm'
# NUM_AUGS = [1, 2, 4, 8]
PUNC_RATIO = [0.10,0.30,0.50]


# Insert punction words into a given sentence with the given ratio "punc_ratio"
def insert_punctuation_marks(sentence, punc_ratio):
    words = sentence.split(' ')
    new_line = []
    q = random.randint(1, int(punc_ratio * len(words) + 1))
    qs = random.sample(range(0, len(words)), q)

    for j, word in enumerate(words):
        if j in qs:
            new_line.append(PUNCTUATIONS[random.randint(0, len(PUNCTUATIONS) - 1)])
            new_line.append(word)
        else:
            new_line.append(word)
    new_line = ' '.join(new_line)
    return new_line


def main(dataset):
    data_df = pd.read_csv(DIR_PATH + '/data/gsm/test_preprocessed.csv')
    for punct in tqdm(PUNC_RATIO):
        data_df[f'noisy_questions'] = data_df['question'].progress_apply(lambda x: insert_punctuation_marks(x,punct))
        data_df.to_csv(f'{DIR_PATH}/data/noisy_datasets/gsm8k_test_noisy_punct_{int(punct*100)}.csv',index=False)
if __name__ == "__main__":
    #for dataset in DATASETS:
    main(DATASET)
