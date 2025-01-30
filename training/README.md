# Tasks

## 1. Text

Objective : 📝 Detecting climate disinformation 📝: based on text from news articles.

| Dataset | nb observations | Task | nb classes | Metrics | Baseline |
|---|---|---|---|---|---|
| text | 6,091 | text classification | 8-class | accuracy | 12,5% |


### Dataset exploration:
- class balancing

### Preprocessing options:
**ML NLP:**
- pre-cleaning : strip, split, lowercase, numbers, punctuation/symbols
- tokenizing | NLTK.tokenize
- stopwords : irrelevant frequent words | nltk.corpus stopwords
- lemmatizing : roots | nltk.stem - WordNetLemmatizer
- vectorizing
    - Bag-of-words (CountVectorizer) counting : word order, doc len, doc context not taken into account
    - Tf-idf (TdidfVectorizer) weighting : word order, doc context not taken into account
        - min_df (infrequent words)
        - max_df (frequent words)
        - max_features (curse of dimensionality)
        - ngram_range = (min_n, max_n) (capturing the context of the words)

**DL NLP:**
- ?pre-cleaning : strip, split, lowercase, numbers, punctuation/symbols
- tokenizing | tensorflow.keras.preprocessing.text.Tokenizer
- padding | tensorflow.keras.preprocessing.sequence.pad_sequences
- Embedding :
    - (layers.Embedding), Word2Vec, FastText

### Baselines models options:
- LLM, finetuned LLM, SLM
    - model needs to be available locally : open source and not too big
    - model needs to be recent and highly performant/ popular
- transformers : BERT, RoBERTa, ALBERT, XLNet, Electra, DeBERTa, XLM-R
- DL NLP : RNN, **LSTM**, CNN, CNN-LSTM, GRU, ANN
- ML NLP : **Naive Bayes**, SVM, Decision Trees, Random Forest, KNN
(- NLI)
(- graph based model)


Idées LLM:
|Nom | Open source | Type | Paramètres | Poids (GB) | Performance | Librairie/Source|
|---|---|---|---|---|---|---|
|Llama 3.1 | Oui | LLM | 8B - 405B | ~16 - 800 | Très bonne | HuggingFace|
|Llama 2 70B | Oui | LLM | 70B | ~140 | Bonne | HuggingFace|
|Falcon 180B | Oui | LLM | 180B | ~360 | Très bonne | HuggingFace|
|Mixtral 8X7B | Oui | LLM | 46,7B | ~93 | Très bonne | HuggingFace|
|Qwen 2.5 (72B) | Oui | LLM | 72B | ~144 | Bonne | HuggingFace|
|Mistral 7B | Oui | SLM | 7,3B | ~14 | Bonne | HuggingFace|
|GPT4All | Oui | SLM | 3-8B | ~3-8 | Moyenne | GPT4All|
|Qwen 2.5 (0.5B) | Oui | SLM | 0,5B | ~1 | Moyenne | HuggingFace|
|Phi-2 | Oui | SLM | 2,7B | ~5 | Bonne | HuggingFace|


Comparison ML and DL:
- MultinomialNB
- BiLSTM

Comparison with traditionnal BERT:
- 

Comparison 3 LLM:

Mistral, Phi-3, Qwen

| Model | Parameters | Quantization Support | Energy Notes | Key Features | 
| ---| ---| ---| ---| ---| 
| Mistral-7B-Instruct | 7B | 4-bit (8GB VRAM) | Efficient architecture, ~5W/inference | Top performance in 7B | class, ideal for instruction-based classification| 
| Phi-3-mini-128k | 3.8B | 4-bit (4GB VRAM) | Ultra-low ~3W/inference | Microsoft's SOTA for small GPUs, long context (128k tokens)| 
| Gemma-2B-it | 2B | 8-bit (3GB VRAM) | Apple-optimized (~2W) | Google's lightweight model, native Metal/MLX support for M3| 
| Llama-3-8B-Instruct | 8B | 4-bit (6GB VRAM) | Optimized KV cache (~7W) | Meta's newest, strong zero-shot classification via system prompts| 
| Qwen1.5-1.8B-Chat | 1.8B | None needed | ~2.5W sustained | Alibaba's efficient model, built-in safety features| 


Lora adapter:

### Optimizations options:
Energy consumption optimizations:
- quantization
- Knowledge distillation

Performance optimization:
- GridSearch for hyperparameters
- ensemble models
- hybrid models (CNN-LSTM)
(langchain)

note : Use Pipelines


LLM prompt
prompt = """
Which category is this sentence ?
\n\n
Respond only with the index of the category
0_not_relevant
1_not_happening
2_not_human
3_not_bad
4_solutions_harmful_unnecessary
5_science_unreliable
6_proponents_biased
7_fossil_fuels_needed
\n\n
sentence :
For most of the Holocene (last 10k years), sea level has been rising at a rate of around 2mm per year. Climate change has little to do with it - it's simply a long term inevitability which will end when the current interglacial returns to a glacial period (which we will regret). 
"""


# CodeCarbon Metrics Table

| **Metric Name**                  | **Metric Unit**  | **Definition**                                | **Formula**                                           | **Usage**                                                  | **Objective**                              |
|-----------------------------------|-----------------|----------------------------------------------|-------------------------------------------------------|------------------------------------------------------------|--------------------------------------------|
| **Carbon Emissions**              | `kgCO₂eq`       | Total CO₂ emissions from computation        | Estimated from energy consumption × carbon intensity  | `tracker.stop()`                                           | Reduce carbon footprint of ML models      |
| **Energy Consumption**            | `kWh`           | Total energy used in kilowatt-hours         | `P × T / 1000` (`P = Power (W)`, `T = Time (s)`)      | `tracker.final_energy.kWh`                                 | Optimize energy efficiency                 |
| **Power Consumption**             | `W` (Watts)     | Average power consumption                   | `E / T` (`E = Energy (J)`, `T = Time (s)`)            | `tracker.final_power.W`                                     | Compare hardware efficiency                |
| **Execution Time**                | `s` (Seconds)   | Total time taken for model inference        | `end_time - start_time`                              | `time.time()` before & after model inference               | Optimize model speed                       |
| **Energy per Prediction**         | `kWh/sample`    | Energy used per sample prediction          | `Total Energy / Number of Predictions`               | `energy_kWh / X_test.shape[0]`                             | Reduce energy usage per inference         |
| **Power per Prediction**          | `W/sample`      | Power used per prediction                   | `Total Power / Number of Predictions`                | `power_W / X_test.shape[0]`                                | Optimize per-sample efficiency             |
| **Carbon Intensity of Grid**      | `gCO₂/kWh`      | CO₂ emissions per kWh of electricity used   | Estimated from regional energy sources               | `tracker._get_carbon_intensity()`                          | Choose low-carbon cloud/compute providers |
