# **FT3**

Fine-tuning Multilingual Large Language Models (LLMs)

---

## **Features**

- **Datasets & Benchmarking**  
  Tools and utilities for managing datasets and evaluating model performance.
- **Inference**  
  Generate predictions using fine-tuned multilingual LLMs.
- **Training**  
  Fine-tune multilingual LLMs on custom datasets.
- **Evaluation**  
  Assess model performance with robust metrics.
- **Gradio Integration**  
  Interactive demos for showcasing model capabilities.

---

## **Benchmark Categories**

- **Code Switching**  
  Handle multilingual scenarios with seamless transitions between languages.
- **Long Conversations**  
  Evaluate the ability to manage extended interactions.
- **Multilingual Support**  
  Test proficiency across multiple languages.

---

## **Tips for Fine-Tuning and Metrics**

- **Token Length in Training**  
  Training on datasets with longer tokens can encourage the LLM to generate longer responses under certain conditions.
  
- **Impact of Shorter Generated Answers on Metrics**  
  When an LLM's output is shorter than the reference, the following metric behaviors are observed:
  - **BLEU-2/4**: Lower scores due to fewer matching n-grams between the shorter output and the reference.
  - **ROUGE-2/L**: Reduced scores as fewer overlapping bigrams (ROUGE-2) or longest common subsequences (ROUGE-L) exist.
  - **BERTScore**: May still show reasonable results as it emphasizes semantic similarity, but shorter outputs can lead to slight reductions in alignment.

---

## **References**

- [FT2 Repository](https://github.com/locchh/ft2)  
- [Shidowake on Hugging Face](https://huggingface.co/shidowake)  
- [Freedom Intelligence on Hugging Face](https://huggingface.co/FreedomIntelligence)  
- [Longquan on Hugging Face](https://huggingface.co/longquan)