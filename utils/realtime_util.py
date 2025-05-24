import pandas as pd
import re

# from configs.api_keys import OPEN_AI

# metrics 
def edit_distance_similarity(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j  # If str1 is empty, insert all characters of str2
            elif j == 0:
                dp[i][j] = i  # If str2 is empty, remove all characters of str1
            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # If last characters are the same, ignore them
            else:
                dp[i][j] = 1 + min(dp[i][j - 1],  # Insert
                                   dp[i - 1][j],  # Remove
                                   dp[i - 1][j - 1])  # Replace

    edit_dist = dp[m][n]
    max_len = max(m, n)
    similarity = 1 - (edit_dist / max_len)

    return similarity

def edit_distance(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j  # If str1 is empty, insert all characters of str2
            elif j == 0:
                dp[i][j] = i  # If str2 is empty, remove all characters of str1
            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # If last characters are the same, ignore them
            else:
                dp[i][j] = 1 + min(dp[i][j - 1],  # Insert
                                   dp[i - 1][j],  # Remove
                                   dp[i - 1][j - 1])  # Replace
    return dp[m][n]

def word_error_rate(reference, hypothesis):
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    edit_dist = edit_distance(reference_words, hypothesis_words)
    return edit_dist / len(reference_words)


def character_error_rate(reference, hypothesis):
    reference_chars = list(reference)
    hypothesis_chars = list(hypothesis)
    edit_dist = edit_distance(reference_chars, hypothesis_chars)
    return edit_dist / len(reference_chars)

def filter_string(input_string):
    # Convert to lowercase and remove newlines
    input_string = input_string.lower().replace('\n', ' ')
    # Keep only lowercase letters, dots, commas, and spaces
    filtered_string = re.sub(r'[^a-z., ]', '', input_string)
    return filtered_string

def remove_consecutive_letters(s):
    if not s:
        return s
    
    result = [s[0]]  # Start with the first character
    for char in s[1:]:
        if char != result[-1]:  # Only add if it's different from the last character
            result.append(char)
    
    return result


# Function to filter the data based on the thresholds
def filter_by_threshold(csv_file, threshold, active_threshold):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Define the columns that correspond to probabilities (a to backspace)
    prob_columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'comma', 'period', 'space', 'backspace']
    
    # Iterate through the DataFrame and filter based on thresholds
    filtered_rows = []
    for index, row in df.iterrows():
        # Get the probabilities for the classes (from a to backspace)
        probabilities = row[prob_columns].values
        
        # Find the maximum probability and its index (class)
        max_prob = max(probabilities)
        max_class = prob_columns[probabilities.argmax()]
        
        # Check if the maximum probability exceeds the threshold and if Active Prob is greater than the active_threshold
        if max_prob > threshold and row['Active Prob'] > active_threshold:
            filtered_rows.append(row)
    
    # Create a DataFrame from the filtered rows
    filtered_df = pd.DataFrame(filtered_rows)
    
    return filtered_df

id2label = [
"a", "b", "c", "d", "e", 
"f", "g", "h", "i", "j", 
"k", "l", "m", "n", "o", 
"p", "q", "r", "s", "t", 
"u", "v", "w", "x", "y", "z", 
"comma", "period", "space", "backspace",
"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
]

label2id = {label: idx for idx, label in enumerate(id2label)}
            
def process_prediction(result_csv, active_thres = 0.5, key_thres = 0.9):
    result = filter_by_threshold(result_csv, key_thres, active_thres)
    # mask = (result['Active Prob'] > active_thres) & (result['Key prediction'] > key_thres)
    chars = result['Key prediction'].tolist()
    processed = []

    i = 0
    while i < len(chars):
        char = chars[i]
        if char == 'period':
            processed.append('.')
        elif char == 'comma': 
            processed.append(',')
        elif char == 'space':
            processed.append(' ')
        elif char == 'backspace':
            if len(processed):
                processed.pop()
        else:
            processed.append(char)
   
        i += 1
    
    # prediction = filter_string(''.join(processed))
    # prediction = remove_consecutive_letters(prediction)
    return processed

def evaluate(prediction, gt):
    prediction = filter_string(prediction)
    print("Corrected: ", prediction)
    print("Ground truth: ", gt)
    print("Edit distance similarity:", edit_distance_similarity(prediction, gt))
    print("Word error rate:", word_error_rate(gt, prediction))

    return edit_distance_similarity(prediction, gt), word_error_rate(gt, prediction)
    

def fix_text(text):
    client = OpenAI(
        api_key=OPEN_AI
    )
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        seed=0,
        messages=[
            {"role": "user", "content": f"Please correct this text, reply in one line:\n {text}. "}
        ]
    )

    m = completion.choices[0].message.content
    print(m)
    return filter_string(m)
