######### WIP  -DO A WANDBHANDLER CLASS

# @title Install wandb
#!pip install wandb
import wandb
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    )


# @title Initialize wandb session
def wandb_init(api_key = None):
    wandb.login(key=api_key)

# @title Start a new evaluation run
config={
"model_id": model_id,
"train_name": train_name,
}

config={
    "model": model_name,
    "type" : 'SLM_FT',
    "tags" : ['baseline', 'FT', 'SLM'],
    "dataset": "QuotaClimat/frugalaichallenge-text-train",
    "prompt" : dataset_train['text'][0],
    "sequence_length": max_seq_length,

    "batch_size": training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
    "max_steps" : training_args.max_steps,
    "epochs" : training_args.num_train_epochs,
    "learning_rate": training_args.learning_rate,
    "warmup_steps": training_args.warmup_steps,

}

def wandb_start_run(
      project="class_LLM_FT",
      group='baseline',
      job_type=['evaluation', 'fine-tuning'], 
      tags=["evaluation", 'test'],
      verbose = False,
      config = None, 
      ):
    if verbose is True:
        for k, v in config.items():
            print(f"{k:{15}} : {v}")
        print()

    return wandb.init(
        project=project,
        group=group,
        tags=tags,
        job_type=job_type,
        config=config
    )




#### EVALUATION
####### FINE TUNED MODEL

# @title Download a specific version of the model artifact from wandb
def wandb_download_model(model_id)
    artifact = run.use_artifact(model_id, type='model')
    model_dir = artifact.download()
    return model_dir

#### EVALUATION
####### BASE MODEL

def wandb_log_artifact_results(results):
  # 1. Create the artifact first
  eval_artifact = wandb.Artifact(
     name="evaluation_results",
     type="evaluation",
     description="Evaluation results on test set"
  )

  # 2. Save the raw results data to a file and add to artifact
  results_df = results.to_pandas()
  results_df.to_csv("test_results.csv", index=False)
  eval_artifact.add_file("test_results.csv")

  # 3. Add metadata to the artifact itself (optional but good practice)
  accuracy = accuracy_score(results['label_true'], results['label_pred'])
  macro_f1 = round(f1_score(results['label_true'], results['label_pred'], average='macro', zero_division=0), 2)

  eval_artifact.metadata = {
     "accuracy": accuracy,
     "macro_f1": macro_f1,
     "num_samples": len(results)
  }

  # 4. Log the artifact
  self.run.log_artifact(eval_artifact)

  return eval_artifact



def generate_metrics_func(results_test, tracker):

    def generate_metrics(results, tracker):
        # 6. Separately log metrics/tables for the run dashboard
        # These appear in charts/visualizations but don't affect artifact content

        accuracy = accuracy_score(results['label_true'], results['label_pred'])
        macro_f1 = round(f1_score(results['label_true'], results['label_pred'], average='macro', zero_division=0), 2)
        efficiency_metrics = tracker.get_metrics()
        results_df = results.to_pandas()
        category_df = category_metrics(results['label_true'], results['label_pred'])

        return {
            "model": config.base_model_name_or_path,
            "eval_accuracy": accuracy,
            "eval_macro_f1": macro_f1,
            "eval_samples": len(results),
            "eval_C02eq": efficiency_df['Emissions (CO2eq)']['inference'],
            "eval_kWh": efficiency_df['Energy (kWh)']['inference'],
            "eval_runtime": efficiency_df['Timings (seconds)']['inference'],
            "eval_results": wandb.Table(dataframe=results_df),
            "eval_scores": wandb.Table(dataframe=category_df),
            "run_efficiency": wandb.Table(dataframe=efficiency_df),
        }
    
    return generate_metrics

def wandb_log_metrics(generate_metrics_func):
    metrics = generate_metrics_func()
    run.log({"eval_dataset": 'test'})
    run.log(metrics)
    return metrics

def wandb_stop():
    self.run.finish()


if __name__ == '__main__':
  
    wandb_init()
    run = wandb_start_run(model_id, train_name)
    results = wandb_log_artifact_results(results_test, artifact)
    metrics = wandb_log_metrics()
    wandb_stop()