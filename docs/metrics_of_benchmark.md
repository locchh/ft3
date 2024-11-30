The choice of metrics depends on the type of task and the evaluation goal. Here's a breakdown of when to use each:

### 1. **BLEU (Bilingual Evaluation Understudy)**
   - **Use BLEU (e.g., BLEU-2, BLEU-4):**
     - For **machine translation** or tasks where precise n-gram matching is important.
     - BLEU-2 measures **bigrams** (two-word sequences), capturing some local context.
     - BLEU-4 measures **4-grams**, assessing longer coherence but might penalize creative text generation.
   - **Considerations:**
     - Works best for short, structured text.
     - Struggles with paraphrasing or semantically equivalent but lexically different outputs.

### 2. **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**
   - **Use ROUGE-2 or ROUGE-4:**
     - For **summarization** tasks.
     - ROUGE-2 measures overlap of bigrams, suitable for short, concise summaries.
     - ROUGE-4 captures more distant dependencies (less common).
   - **Use ROUGE-L:**
     - For assessing **longer sequences** or text with semantic alignment.
     - It measures the **Longest Common Subsequence (LCS)**, accounting for sentence structure and order.

### 3. **BERTScore**
   - **Use BERTScore:**
     - For **semantic similarity** tasks where the meaning is more critical than exact word matching.
     - Suitable for **creative tasks** (e.g., paraphrasing, open-ended generation) where traditional n-gram metrics fail.
   - **Advantages:**
     - Leverages contextual embeddings from BERT, making it robust to synonyms and rephrasings.

### **Summary of Recommendations**
- **Machine Translation:** BLEU (BLEU-4 for fluency and BLEU-2 for shorter phrases).
- **Text Summarization:** ROUGE-2 or ROUGE-L (ROUGE-2 for short summaries, ROUGE-L for structural alignment).
- **Paraphrasing / Open-Ended Generation:** BERTScore (semantic evaluation).
- **General Multi-Metric Evaluation:** Use a combination (e.g., BLEU, ROUGE, and BERTScore) to balance lexical and semantic quality.

Would you like more details about any metric or its implementation?