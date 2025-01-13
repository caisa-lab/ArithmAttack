import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import DIR_PATH

PUNCTUATIONS = ['.', ',', '!', '?', ';', ':']


'''
, My name is : Shahzeb [,, My, name , is, :, Shahzeb]
My name is Shahzeb [My, name , is, Shahzeb]

'''



def calculate_punctuations(original_question, noisy_question):
    original_words = original_question.split(' ')
    noisy_words = noisy_question.split(' ')
    
    punctuation_count = 0
    insertion_positions = []
    
    original_index = 0
    noisy_index = 0

    while original_index < len(original_words) and noisy_index < len(noisy_words):
        if original_words[original_index] == noisy_words[noisy_index]:
            original_index += 1
            noisy_index += 1
        else:
            if noisy_words[noisy_index][0] in PUNCTUATIONS:
                punctuation_count += 1
                insertion_positions.append(noisy_index)
                noisy_index += 1
            else:
                original_index += 1
                noisy_index += 1
    return punctuation_count, insertion_positions

def process_dataframes(original_df, noisy_dfs):
    # Create a combined DataFrame for visualization
    results = []

    for df in noisy_dfs:
        for index, row in df.iterrows():
            original_question = original_df.loc[index, 'Question']
            noisy_question = row['Question']
            llm_answer = row['Answer - LLM Converted']
            ground_truth_answer = row['Answer - Ground Truth Converted']
            
            # Calculate punctuations and insertion positions
            punctuation_count, insertion_positions = calculate_punctuations(original_question, noisy_question)
            
            # Determine if the LLM answer is correct
            is_correct = llm_answer == ground_truth_answer
            
            # Append the result to a list
            results.append({
                'punctuation_count': punctuation_count,
                'insertion_positions': insertion_positions,
                'is_correct': is_correct
            })
            # print(noisy_question,punctuation_count,insertion_positions)
    
    # Convert results into a DataFrame
    return pd.DataFrame(results)


def visualize_results(results_df,model_name, bin_size=10):
    # Map 'is_correct' to 'Correct' and 'Incorrect'
    results_df['is_correct'] = results_df['is_correct'].map({True: 'Correct', False: 'Incorrect'})
    
    # Create bins for punctuation_count based on the specified bin_size, ensuring no negative values
    max_punctuation_count = results_df['punctuation_count'].max()
    bins = range(0, max_punctuation_count, bin_size)
    results_df['punctuation_bin'] = pd.cut(results_df['punctuation_count'], bins=bins, right=False)

    # Group by bins and calculate total correct and incorrect in each bin
    binned_punctuation_df = results_df.groupby('punctuation_bin').agg({
        'is_correct': lambda x: (x == 'Correct').sum(),
        'punctuation_count': 'count'
    }).reset_index()
    
    # Separate correct and incorrect counts
    binned_punctuation_df['incorrect'] = binned_punctuation_df['punctuation_count'] - binned_punctuation_df['is_correct']
    binned_punctuation_df.rename(columns={'is_correct': 'correct'}, inplace=True)

    # Set up the figure size
    plt.figure(figsize=(12, 8))

    # Plot the 'correct' portion first
    sns.barplot(x='punctuation_bin', y='correct', data=binned_punctuation_df, color='green', label='Correct')

    # Overlay the 'incorrect' portion on top of the 'correct'
    sns.barplot(x='punctuation_bin', y='incorrect', data=binned_punctuation_df, 
                bottom=binned_punctuation_df['correct'], color='red', label='Incorrect')

    # Add labels and title
    plt.xlabel('Punctuation Count')
    plt.ylabel('Count of Answers')
    plt.title('Correct and Incorrect Counts grouped by Punctuation Count')

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45)

    # Add a legend to differentiate between correct and incorrect
    plt.legend(title='Answer')

    # Show the plot
    plt.tight_layout()
    plt.savefig(f'multiArith_{model_name}_punctuation_count_stacked.pdf')  # Save plot
    plt.close()

def visualize_results_with_positions(results_df, model_name,bin_size=10):
    # Step 1: Flatten the insertion positions for better visualization
    flat_positions = dict()
    
    # Iterate through each row in the DataFrame
    for _, row in results_df.iterrows():
        for pos in row['insertion_positions']:
            if pos not in flat_positions:
                # Initialize counts for correct and incorrect
                flat_positions[pos] = {'correct': 0, 'incorrect': 0}
            
            # Increment correct or incorrect count based on the 'is_correct' column
            if row['is_correct']:
                flat_positions[pos]['correct'] += 1
            else:
                flat_positions[pos]['incorrect'] += 1

    # Convert flat_positions dictionary to a DataFrame
    flat_positions_df = pd.DataFrame([
        {'insertion_points': pos, 'correct': counts['correct'], 'incorrect': counts['incorrect']}
        for pos, counts in flat_positions.items()
    ])

    # Create bins for insertion_points based on the specified bin_size
    max_insertion_point = flat_positions_df['insertion_points'].max()
    bins = range(0, max_insertion_point, bin_size)
    flat_positions_df['insertion_point_bin'] = pd.cut(flat_positions_df['insertion_points'], bins=bins, right=False)

    # Group by bins and calculate total correct and incorrect in each bin
    binned_positions_df = flat_positions_df.groupby('insertion_point_bin').agg({'correct': 'sum', 'incorrect': 'sum'}).reset_index()

    # Melt the DataFrame to get it in a long format suitable for seaborn's barplot
    binned_positions_melted = binned_positions_df.melt(id_vars='insertion_point_bin', 
                                                       value_vars=['correct', 'incorrect'], 
                                                       var_name='Answer', 
                                                       value_name='Count')

    # Set up the figure size
    plt.figure(figsize=(12, 8))
    # Plot the 'correct' portion first
    sns.barplot(x='insertion_point_bin', y='correct', data=binned_positions_df, color='green', label='Correct')

    # Overlay the 'incorrect' portion on top of the 'correct'
    sns.barplot(x='insertion_point_bin', y='incorrect', data=binned_positions_df, 
                bottom=binned_positions_df['correct'], color='red', label='Incorrect')

    # Add labels and title
    plt.xlabel('Insertion Points')
    plt.ylabel('Count of Answers')
    plt.title('Correct and Incorrect Counts grouped by Insertion Points')

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45)

    # Add a legend to differentiate between correct and incorrect
    plt.legend(title='Answer')

    # Show the plot
    plt.tight_layout()
    plt.savefig(f'multiArith_{model_name}_punctuation_insertion_distribution_stacked.pdf')  # Save plot
    plt.close()

all_dir_names = ['Meta-Llama-3-8B-Instruct','Mathstral-7b-v0.1','Mistral-7B-Instruct-v0.2','Llama-3.1-8B-Instruct','gemma-2-2b-it','zephyr-7b-beta','Qwen2.5-1.5B-Instruct']


for dir_name in all_dir_names:
    clean_file_path = f"{DIR_PATH}/data/multiArith/{dir_name}/{dir_name}_converted.csv"
    attacked_file_path_converted_10 = f"{DIR_PATH}/data/multiArith/{dir_name}/{dir_name}_noisy_new_10_converted.csv"
    attacked_file_path_converted_30 = f"{DIR_PATH}/data/multiArith/{dir_name}/{dir_name}_noisy_new_30_converted.csv"
    attacked_file_path_converted_50 = f"{DIR_PATH}/data/multiArith/{dir_name}/{dir_name}_noisy_new_50_converted.csv"
    df_clean = pd.read_csv(clean_file_path)
    df_10per = pd.read_csv(attacked_file_path_converted_10)
    df_30per = pd.read_csv(attacked_file_path_converted_30)
    df_50per = pd.read_csv(attacked_file_path_converted_50)

    # Assume original_df and noisy_df1, noisy_df2, noisy_df3 are your dataframes
    noisy_dfs = [df_10per, df_30per, df_50per]

    # Process dataframes to calculate punctuation counts and correctness
    results_df = process_dataframes(df_clean, noisy_dfs)

    # Visualize the results
    visualize_results_with_positions(results_df,dir_name,bin_size=5)
    visualize_results(results_df,dir_name,bin_size=5)
    