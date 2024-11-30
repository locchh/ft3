# ft3

fine tune multilingual LLM

# features

- datasets, benchmark
- inference
- train
- evaluate
- gradio

# tips

- Training on datasets with longer tokens can influence an LLM (large language model) to generate more tokens under certain conditions

- If the LLM's generated answer is shorter than the reference, the following happens to the metrics:

    BLEU-2/4: Scores decrease because fewer n-grams match between the shorter candidate and the reference.

    ROUGE-2/L: Scores decrease as fewer overlapping bigrams (ROUGE-2) or longest common subsequences (ROUGE-L) exist between the candidate and reference.

    BERTScore: Scores might still remain reasonable because BERTScore considers semantic similarity, but shorter outputs may still lead to slight reductions due to incomplete alignment with the reference.

# references

[ft2](https://github.com/locchh/ft2)

[shidowake](https://huggingface.co/shidowake)

[FreedomIntelligence](https://huggingface.co/FreedomIntelligence)

[longquan](https://huggingface.co/longquan)