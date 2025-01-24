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
