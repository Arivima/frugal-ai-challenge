# Frugal AI Challenge

🌎 Find all the details of the challenge at https://frugalaichallenge.org/

[session info](https://www.youtube.com/watch?v=loxJmDaN-zI)
https://www.elysee.fr/en/sommet-pour-l-action-sur-l-ia
- AI Action Summit Paris 2025
    - Feb 6-7 : Science days
    - Feb 8-9 : Cultural events
    - Feb 10 : Forum (conferences & workshop)
    - Feb 11 : Leaders' session and side events
- Evaluation
    - matrix accuracyXEnergy:
        - at least 90% accuracy to be considered
        - Bottom 10 energy consumption
    - final selection based on real world application 
        - model cards : approach and training
        - other metrics : minimize false positives precision recall ?
    - hidden test set will be run on Nvidia T4 small GPU (smallest on HF)
    - LLM interesting use case
- Ideas:
    - each run (model_name_timestamp_M1_M2) saved + metrics -> on plotly matrix
    - try different baseline models
    - notebook with exploratory analysis

- consent for publishing on this

[Challenge repository on Github](git@github.com:frugal-ai-challenge/frugal-ai-challenge.github.io.git)

[Challenge space on HuggingFace](https://huggingface.co/frugal-ai-challenge)

## Tasks

- 📝 Detecting climate disinformation 📝: based on text from news articles.
- 🔥 Classifying regions at risk of wildfires 🔥: based on image data gathered via satellite.
- 🌳🪓 Detecting illegal deforestation 🌳🪓 : from bio-acoustic data recorded in the jungle


| Dataset | observations | Task | Metrics | Baseline |
|---|---|---|---|---|
| text | 6,091 | classification 8-class| accuracy | 12,5% |
| audio | 50,397 | classification binary  |accuracy | 50% |
| images | 33,636 | classification binary  |accuracy, IOU | 50% | 

## Datasets

3 datasets for the Frugal AI Challenge :

**Images** - wildfire detection - 3.43 GB

```
git clone https://huggingface.co/datasets/pyronear/pyro-sdis
```
```
from datasets import load_dataset

ds = load_dataset("pyronear/pyro-sdis")
```

**Audio** - Chainsaw detection - 3.78 GB
```
git clone https://huggingface.co/datasets/rfcx/frugalai
```
```
from datasets import load_dataset

ds = load_dataset("rfcx/frugalai")
```

**Text** - Climate disinformation classification - 1.21 MB
```
git clone https://huggingface.co/datasets/QuotaClimat/frugalaichallenge-text-train
```
```
from datasets import load_dataset

ds = load_dataset("QuotaClimat/frugalaichallenge-text-train")
```
```
import pandas as pd

df = pd.read_parquet("hf://datasets/QuotaClimat/frugalaichallenge-text-train/train.parquet")
```

## Submission

**To submit a model :**
- on HF: duplicate a templated HF API for each model that needs to be submitted [submission-template](https://huggingface.co/spaces/frugal-ai-challenge/submission-template) (2 vCPU 16 GB RAM)
- locally : train a model
- on GCP : upload model to a gcs
- on HF : update the relevant task with load_model() and model.predict()
- test with Swagger https://arivima-submission-template.hf.space/docs/
- submit model to leaderboard [submission-portal](https://huggingface.co/spaces/frugal-ai-challenge/submission-portal)

leaderboards:
- Text - https://huggingface.co/datasets/frugal-ai-challenge/public-leaderboard-text
- Image - https://huggingface.co/datasets/frugal-ai-challenge/public-leaderboard-image
- Audio - https://huggingface.co/datasets/frugal-ai-challenge/public-leaderboard-audio

**The API :**  
- takes a dataset config as input
```
{
  "test_size": 0.2,
  "test_seed": 42,
  "dataset_name": "QuotaClimat/frugalaichallenge-text-train"
}
```
- outputs the metrics of the submitted model
```
{
  "username": "Arivima",
  "space_url": "https://huggingface.co/spaces/Arivima/submission-template",
  "submission_timestamp": "2025-01-22T12:48:04.456838",
  "model_description": "Random Baseline",
  "accuracy": 0.13453650533223954,
  "energy_consumed_wh": 0.0007554576981390744,
  "emissions_gco2eq": 0.00027886540742533766,
  "emissions_data": {
    "run_id": "1216029c-6cf4-4dba-bcb0-211153335710",
    "duration": 0.012289291946217418,
    "emissions": 2.7886540742533767e-7,
    "emissions_rate": 0.000019623978327623796,
    "cpu_power": 150,
    "gpu_power": 0,
    "ram_power": 46.42727851867676,
    "cpu_energy": 5.865745382228245e-7,
    "gpu_energy": 0,
    "ram_energy": 1.6888315991624992e-7,
    "energy_consumed": 7.554576981390744e-7,
    "country_name": "United States",
    "country_iso_code": "USA",
    "region": "virginia",
    "cloud_provider": "",
    "cloud_region": "",
    "os": "Linux-5.10.230-223.885.amzn2.x86_64-x86_64-with-glibc2.36",
    "python_version": "3.9.21",
    "codecarbon_version": "2.8.2",
    "cpu_count": 16,
    "cpu_model": "Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz",
    "gpu_count": null,
    "gpu_model": null,
    "ram_total_size": 123.80607604980469,
    "tracking_mode": "machine",
    "on_cloud": "N",
    "pue": 1
  },
  "api_route": "/text",
  "dataset_config": {
    "dataset_name": "QuotaClimat/frugalaichallenge-text-train",
    "test_size": 0.2,
    "test_seed": 42
  }
}
```

