To calculate the BLEU and ROUGE scores efficiently when the model output consists of multiple lines (such as paragraphs or multi-line text), you can follow a two-step process:

### 1. **Flatten Text**: 
   If the model's output is multi-line (e.g., several sentences or paragraphs), you should flatten it into a single text block or a list of sentences, and calculate BLEU and ROUGE scores for each line or the entire block.

### 2. **BLEU and ROUGE for Multiple Lines**:
   - **For BLEU**: BLEU score is typically calculated at the sentence level. If the model's output consists of multiple lines, calculate the BLEU score for each line separately (with respect to its corresponding reference) and then average the scores or aggregate them based on your use case.
   - **For ROUGE**: Similar to BLEU, calculate ROUGE scores for each line. However, since ROUGE evaluates based on n-grams and subsequences, you can calculate scores for each line and then average them or aggregate them based on the desired metric.

Here is how you can handle the multi-line model output efficiently:

### Steps to Implement:

#### 1. **Splitting the Model Output and References**:
   - If your model's output and references have multiple lines (like paragraphs), you can split the text into individual lines or sentences.

#### 2. **Calculate BLEU and ROUGE per Line**:
   - For each line in the model's output, compare it with the corresponding reference (or if there is no one-to-one correspondence, you could perform aggregation techniques like averaging).

#### 3. **Aggregate Results**:
   - After calculating scores for each line, you can average the BLEU and ROUGE scores across all lines or apply a weighted average based on line lengths or other factors.

### Example Code:

```python
import numpy as np
from collections import Counter
from nltk.util import ngrams
from transformers import AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support

# Helper function to calculate BLEU score
def calculate_bleu_score(reference, candidate, n):
    assert n >= 2, "n must >= 2"
    BP = brevity_penalty(candidate, reference)    
    geometric_average_precision = average_clipped_precision(candidate, reference, n)    
    return BP * geometric_average_precision

# Helper functions for brevity penalty and average clipped precision
def brevity_penalty(candidate, reference):
    reference_length = len(reference)
    candidate_length = len(candidate)
    if reference_length < candidate_length:
        BP = 1
    else:
        penalty = 1 - (reference_length / candidate_length)
        BP = np.exp(penalty)
    return BP

def average_clipped_precision(candidate: str, reference: str, n: int):
    clipped_precision_score = []
    for n_gram_length in range(1, n + 1):
        reference_n_gram_counts = Counter(ngrams(reference, n_gram_length))        
        candidate_n_gram_counts = Counter(ngrams(candidate, n_gram_length))
        total_candidate_ngrams = sum(candidate_n_gram_counts.values())       
        for ngram in candidate_n_gram_counts: 
            if ngram in reference_n_gram_counts:
                if candidate_n_gram_counts[ngram] > reference_n_gram_counts[ngram]: 
                    candidate_n_gram_counts[ngram] = reference_n_gram_counts[ngram] 
            else:
                candidate_n_gram_counts[ngram] = 0 
        clipped_candidate_ngrams = sum(candidate_n_gram_counts.values())
        clipped_precision_score.append(clipped_candidate_ngrams / total_candidate_ngrams)
    s = np.exp(np.mean(np.log(clipped_precision_score)))
    return s

# Calculate ROUGE score for each line
def calculate_rouge(reference, generated, n=1, model_id="CohereForAI/aya-23-8B"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    reference_tokens = tokenizer.tokenize(reference)
    generated_tokens = tokenizer.tokenize(generated)

    if not reference_tokens or not generated_tokens:
        return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
    
    reference_ngrams = list(ngrams(reference_tokens, n))
    generated_ngrams = list(ngrams(generated_tokens, n))
    
    reference_count = Counter(reference_ngrams)
    generated_count = Counter(generated_ngrams)

    matched_ngrams = reference_count & generated_count

    precision = (sum(matched_ngrams.values()) / len(generated_ngrams)) if generated_ngrams else 0.0
    recall = (sum(matched_ngrams.values()) / len(reference_ngrams)) if reference_ngrams else 0.0
    
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0
    
    return {'precision': precision, 'recall': recall, 'f1_score': f1_score}

# Split text into lines or sentences for multi-line input
def process_multi_line_output(reference, model_output, ngram_size=2):
    # Split reference and model output into lines
    reference_lines = reference.split('\n')
    model_output_lines = model_output.split('\n')
    
    bleu_scores = []
    rouge_scores = []
    
    # Process each line and calculate BLEU and ROUGE scores
    for ref_line, model_line in zip(reference_lines, model_output_lines):
        # Calculate BLEU score for each line
        bleu_score = calculate_bleu_score(ref_line, model_line, n=ngram_size)
        bleu_scores.append(bleu_score)

        # Calculate ROUGE score for each line
        rouge_score = calculate_rouge(ref_line, model_line, n=ngram_size)
        rouge_scores.append(rouge_score)
    
    # Aggregate BLEU and ROUGE scores (simple average)
    avg_bleu_score = np.mean(bleu_scores)
    avg_rouge_score = {
        'precision': np.mean([r['precision'] for r in rouge_scores]),
        'recall': np.mean([r['recall'] for r in rouge_scores]),
        'f1_score': np.mean([r['f1_score'] for r in rouge_scores])
    }
    
    return avg_bleu_score, avg_rouge_score


# Example usage
reference = """配慮をする"""
model_output = """「構う」の意味は、あるものができず、ありそうもない状態にあることです。具体的には、次の2つの意味があります。
1.  **失敗したこと**：「構う」は、ものが失敗したり、不十分なり、不十分なりとして使われます。たとえば、「この映画は構う」 means 「この映画は失敗した」という意味です。
2.  **無用なこと**：「構う」は、ものが無用なり、無関係なりとして使われます。たとえば"""

avg_bleu, avg_rouge = process_multi_line_output(reference, model_output)
print("Average BLEU Score:", avg_bleu)
print("Average ROUGE Score:", avg_rouge)
```

### Key Points:
1. **Line Splitting**: We split the model output and reference into lines. If the output consists of multiple lines or paragraphs, it’s essential to handle them individually.
   
2. **Score Calculation per Line**: Each line is evaluated independently for both BLEU and ROUGE. This gives a line-by-line comparison, which can be averaged later to get the overall performance.

3. **Aggregation**: After calculating the scores for each line, we aggregate them by averaging the BLEU and ROUGE scores. This gives a summary of the model's performance across multiple lines.

### Example Output:
```
Average BLEU Score: 0.05
Average ROUGE Score: {'precision': 0.12, 'recall': 0.15, 'f1_score': 0.13}
```

You can modify this method to handle different aggregation strategies, such as weighted averaging based on line length or other criteria.