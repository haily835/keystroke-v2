import pandas as pd
import re
from openai import OpenAI
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

def process_prediction(result_csv, active_thres = 0.5, key_thres = 0.9):
    result = pd.read_csv(result_csv)
    mask = (result['Active Prob'] > active_thres) & (result['Key Prob'] > key_thres)
    chars = result.loc[mask, 'Key prediction'].tolist()
    processed = []

    i = 0
    while i < len(chars):
        char = chars[i]
        if char == 'dot':
            processed.append('.')
        elif char == 'comma': 
            processed.append(',')
        elif char == 'space':
            processed.append(' ')
        elif char == 'delete':
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
