#JSON TO CSV FILE FROM LLM ANSWERS
#
#
#

import pandas as pd
import json

# Specify the path to your JSON file
json_file_path = '/content/english.json'

try:
    with open(json_file_path, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: JSON file not found at '{json_file_path}'.")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from '{json_file_path}'. Please ensure it's valid JSON.")
    exit()

excel_data = []

for sentence_pair_id, values in data.items():
    row_data = {
        "Sentence Pair ID": sentence_pair_id,
        "Original Tokens Count": values.get("number_of_tokens_in_original_sentence", ""),
        "Non-Penalizable Errors": ", ".join(values.get("non_penalizable_errors", [])),
        "Major Penalizable Errors": ", ".join(values.get("major_penalizable_errors", [])),
        "Minor Penalizable Errors": ", ".join(values.get("minor_penalizable_errors", [])),
        "Total Penalty": values.get("total_penalty", ""),
        "Score": values.get("score", "")
    }
    excel_data.append(row_data)

df = pd.DataFrame(excel_data)

df.to_excel("/content/output.xlsx", index=False)

print("JSON data from file successfully converted to output.xlsx")

#CSV TO ERROR LIST
#
#
#


import pandas as pd

# 1. Load your data
df = pd.read_csv('/content/output-30.csv')  # ← replace with your file path

# 2. Split the target column into lists
#    Replace 'your_column' with the name of the column to split
df['non'] = df['non'].str.split(',')

# 3. Explode so each list‐item becomes its own row
df = df.explode('non')

# 4. Clean up whitespace around each phrase (optional)
df['non'] = df['non'].str.strip()

# 5. Save the result
df.to_csv('output.csv', index=False)
print(f"Expanded from {len(pd.read_csv('/content/output-30.csv'))} to {len(df)} rows and saved to output.csv")


#ERROR LIST TO WORD PAIR LIST
#
#
#


import pandas as pd

# 1. Load your data
df = pd.read_csv('/content/output.csv')  # ← replace with your file path

# 2. Extract A, B, C from strings like "A vs B (C)"
pattern = r'^(.*?)\s+vs\s+(.*?)\s*\((.*)\)\s*$'
df[['A', 'B', 'C']] = df['non'].str.extract(pattern)

# 3. (Optional) Drop the old column
# df = df.drop(columns=['your_column'])

# 4. Clean up whitespace
for col in ['A','B','C']:
    df[col] = df[col].str.strip()

# 5. Save the result
df.to_csv('output.csv', index=False)
print(f"Processed {len(df)} rows and saved to output.csv")

#LIST OF NO-MISMATCH PAIRS RANDOMLY SAMPLED
#
#
#


import pandas as pd
import random

# 1. Load your CSV
# Replace 'input.csv' with your file name, and 'sentence' with your column name.
df = pd.read_csv('/content/Book5.csv')

# 2. For each sentence, sample 2 distinct words
#    and store them as a list in a new column 'picked_words'
def pick_two_words(sent):
    words = sent.split()
    if len(words) < 2:
        # if fewer than 2 words, just repeat or pad as needed
        return words + [''] * (2 - len(words))
    return random.sample(words, 2)

df['picked_words'] = df['Predicted'].apply(pick_two_words)

# 3. Explode so that each word gets its own row
#    This will give you 2× as many rows as the original
df_expanded = df.explode('picked_words').reset_index(drop=True)

# 4. (Optional) Rename the exploded column to something meaningful
df_expanded = df_expanded.rename(columns={'picked_words': 'random_word'})

# 5. Save to a new CSV or continue processing
df_expanded.to_csv('output_with_random_words.csv', index=False)

print(f"Original rows: {len(df)}, Expanded rows: {len(df_expanded)}")


