import numpy  as np
from nltk import ngrams
from collections import Counter
from transformers import AutoTokenizer
from bert_score import score

def convert_to_llama_format(input_list):
    conversion_map = {
        'from': 'role',
        'value': 'content',
        'gpt': 'assistant',
        'human': 'user'
    }
    
    output_list = []
    for entry in input_list:
        converted_entry = {
            conversion_map['from']: conversion_map.get(entry['from'], entry['from']),
            conversion_map['value']: entry['value']
        }
        output_list.append(converted_entry)
    
    return output_list


def calculate_metrics(reference:str, candidate:str):
    bert_score = calcuate_bert(reference, candidate)
    bleu_2_score = calculate_bleu_score(reference, candidate,2)
    bleu_4_score = calculate_bleu_score(reference, candidate,4)
    rouge_2_score = calculate_rouge(reference, candidate, n=2)
    rouge_l_score = calculate_rouge_l(reference, candidate)
    return bert_score["f1_score"], bleu_2_score, bleu_4_score, rouge_2_score["f1_score"], rouge_l_score["f1_score"]

# Calculate BERT
def calcuate_bert(reference:str, candidate:str):
    P, R, F1 = score([candidate], [reference], lang="ja")  # Set language to Japanese
    #print(f"BERTScore: Precision={P.mean():.4f}, Recall={R.mean():.4f}, F1={F1.mean():.4f}")
    return {
        'precision': float(P),
        'recall': float(R),
        'f1_score': float(F1)
    }
    

# Calculate ROUGE, ROUGE-L
def calculate_rouge(reference, generated, n=1, model_id = "meta-llama/Llama-3.2-1B-Instruct"):

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Tokenize the input strings into words
    reference_tokens = tokenizer.tokenize(reference) #reference.split()
    generated_tokens = tokenizer.tokenize(generated) #generated.split()
    
    # Generate n-grams
    reference_ngrams = list(ngrams(reference_tokens, n))
    generated_ngrams = list(ngrams(generated_tokens, n))
    
    # Count n-grams
    reference_count = Counter(reference_ngrams)
    generated_count = Counter(generated_ngrams)

    # Calculate matched n-grams
    matched_ngrams = reference_count & generated_count
    
    # Precision
    precision = (sum(matched_ngrams.values()) / len(generated_ngrams)) if generated_ngrams else 0.0
    
    # Recall
    recall = (sum(matched_ngrams.values()) / len(reference_ngrams)) if reference_ngrams else 0.0
    
    # F1 Score
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

def lcs_length(x, y):
    """Calculate the length of the longest common subsequence (LCS)"""
    m, n = len(x), len(y)
    # Create a 2D array to store lengths of longest common subsequence.
    lcs_table = [[0] * (n + 1) for _ in range(m + 1)]

    # Fill the lcs_table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
            else:
                lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])

    return lcs_table[m][n]

def calculate_rouge_l(reference, generated, model_id = "meta-llama/Llama-3.2-1B-Instruct"):

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Tokenize the input strings into words
    reference_tokens = tokenizer.tokenize(reference) #reference.split()
    generated_tokens = tokenizer.tokenize(generated) #generated.split()

    # Calculate the length of the longest common subsequence
    lcs_len = lcs_length(reference_tokens, generated_tokens)

    # Precision
    precision = lcs_len / len(generated_tokens) if generated_tokens else 0.0

    # Recall
    recall = lcs_len / len(reference_tokens) if reference_tokens else 0.0

    # F1 Score
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

# Calculate BLEU-score
def brevity_penalty(candidate, reference):
    """
    Calculates the brevity penalty given the candidate and reference sentences.
    """
    reference_length = len(reference)
    candidate_length = len(candidate)

    if reference_length < candidate_length:
        BP = 1
    else:
        penalty = 1 - (reference_length / candidate_length)
        BP = np.exp(penalty)

    return BP


def average_clipped_precision(candidate:str, reference:str,n:int):
    """
    Calculates the precision given the candidate and reference sentences.
    """

    clipped_precision_score = []
    
    # Loop through values 1, 2, 3, 4. This is the length of n-grams
    for n_gram_length in range(1, n):
        reference_n_gram_counts = Counter(ngrams(reference, n_gram_length))        
        candidate_n_gram_counts = Counter(ngrams(candidate, n_gram_length))

        total_candidate_ngrams = sum(candidate_n_gram_counts.values())       
        
        for ngram in candidate_n_gram_counts: 
            # check if it is in the reference n-gram
            if ngram in reference_n_gram_counts:
                # if the count of the candidate n-gram is bigger than the corresponding
                # count in the reference n-gram, then set the count of the candidate n-gram 
                # to be equal to the reference n-gram
                
                if candidate_n_gram_counts[ngram] > reference_n_gram_counts[ngram]: 
                    candidate_n_gram_counts[ngram] = reference_n_gram_counts[ngram] # t
                                                   
            else:
                candidate_n_gram_counts[ngram] = 0 # else set the candidate n-gram equal to zero

        clipped_candidate_ngrams = sum(candidate_n_gram_counts.values())
        
        clipped_precision_score.append(clipped_candidate_ngrams / total_candidate_ngrams)
    
    # Calculate the geometric average: take the mean of elemntwise log, then exponentiate
    # This is equivalent to taking the n-th root of the product as shown in equation (1) above
    s = np.exp(np.mean(np.log(clipped_precision_score)))
    
    return s

def calculate_bleu_score(reference:str,candidate:str, n:int):
    assert n >=2, "n must >= 2"
    BP = brevity_penalty(candidate, reference)    
    geometric_average_precision = average_clipped_precision(candidate, reference, n)    
    return BP * geometric_average_precision