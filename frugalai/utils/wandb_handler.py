# wandb_handler

#Prerequisites
    #!pip install wandb

import wandb
import os
from evaluation import category_metrics
from efficiency_tracker import FunctionTracker
from sklearn.metrics import f1_score, accuracy_score
from typing import Optional, Dict, List, Any, Union
import pandas as pd
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class ExperimentConfig:
    """Configuration for a specific experiment type."""
    project: str
    group: str
    job_type: str
    tags: List[str]
    config_template: callable

class ExperimentConfigManager:
    """Manages experiment configurations."""
    
    EXPERIMENT_CONFIGS = {
        'llm_base_evaluation': ExperimentConfig(
            project="class_LLM_FT",
            group='llm_base',
            job_type="evaluation",
            tags=["evaluation", 'test', 'base_model', 'full_run'],
            config_template=lambda model_name: {"model_name": model_name}
        ),
        'llm_ft_evaluation': ExperimentConfig(
            project="class_LLM_FT",
            group='llm_ft',
            job_type="evaluation",
            tags=["evaluation", 'test', 'ft_model', 'full_run'],
            config_template=lambda model_name: {"model": model_name}
        ),
        'llm_ft_classification_head': ExperimentConfig(
            project="class_LLM_FT",
            group='llm_ft_class_head',
            job_type="fine-tuning",
            tags=["fine-tuning", 'train', 'ft_model', '50'],
            config_template=lambda model_name: {"model": model_name}
        ),
        'llm_kd_offline_teacher': ExperimentConfig(
            project="class_LLM_KD",
            group='teacher',
            job_type="generation",
            tags=["kd", 'teacher', '50', 'gen'],
            config_template=lambda model_name: {"model": model_name}
        ),
        'llm_kd_offline_student': ExperimentConfig(
            project="class_LLM_KD",
            group='student',
            job_type="training",
            tags=["kd", 'student', '50', 'train'],
            config_template=lambda model_name: {"model": model_name}
        )
    }

    @classmethod
    def get_config(cls, experiment: str) -> ExperimentConfig:
        """Get configuration for a specific experiment type."""
        if experiment not in cls.EXPERIMENT_CONFIGS:
            raise ValueError(f"Unknown experiment type: {experiment}")
        return cls.EXPERIMENT_CONFIGS[experiment]

class MetricsCalculator:
    """Handles calculation of various metrics."""
    
    @staticmethod
    def calculate_metrics(true_labels: pd.Series, predicted_labels: pd.Series) -> Dict[str, Any]:
        """Calculate accuracy, F1 score and category metrics."""
        if len(true_labels) == 0 or len(predicted_labels) == 0:
            raise ValueError("Input labels cannot be empty")
            
        accuracy = accuracy_score(true_labels, predicted_labels)
        macro_f1 = round(f1_score(true_labels, predicted_labels, 
                                 average='macro', zero_division=0), 2)
        
        try:
            category_df = category_metrics(true_labels, predicted_labels)
            if not isinstance(category_df, pd.DataFrame):
                raise ValueError("category_metrics must return a pandas DataFrame")
        except Exception as e:
            raise RuntimeError(f"Failed to calculate category metrics: {str(e)}")
        
        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "category_df": category_df
        }

class ArtifactManager:
    """Manages wandb artifacts."""
    
    def __init__(self, run: wandb.Run):
        self.run = run

    def upload_model(self, model_path: str, experiment: str) -> wandb.Artifact:
        """Upload a model artifact to wandb."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
            
        name = "model_FT_classif_head" if experiment == 'llm_ft_classification_head' else "model"
        final_artifact = wandb.Artifact(name=name, type="model")
        final_artifact.add_dir(model_path)
        self.run.log_artifact(final_artifact)
        return final_artifact

    def download_model(self, model_id: str) -> str:
        """Download a specific version of the model artifact from wandb."""
        try:
            artifact = self.run.use_artifact(model_id, type='model')
            return artifact.download()
        except Exception as e:
            raise RuntimeError(f"Failed to download model artifact: {str(e)}")

    def log_results_artifact(self, results: pd.DataFrame, metrics: Dict[str, Any]) -> wandb.Artifact:
        """Log evaluation results as a wandb artifact."""
        eval_artifact = wandb.Artifact(
            name="evaluation_results",
            type="evaluation",
            description="Evaluation results on test set"
        )
        
        results.to_csv("test_results.csv", index=False)
        eval_artifact.add_file("test_results.csv")
        
        eval_artifact.metadata = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "num_samples": len(results)
        }
        
        self.run.log_artifact(eval_artifact)
        return eval_artifact

class WandbHandler:
    """Main class for managing Weights & Biases operations."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the WandbHandler."""
        try:
            wandb.login(key=api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to login to wandb: {str(e)}")
            
        self.run: Optional[wandb.Run] = None
        self.experiment: Optional[str] = None
        self.results_df: Optional[pd.DataFrame] = None
        self.metrics: Optional[Dict[str, Any]] = None
        self.artifact_manager: Optional[ArtifactManager] = None

    def start_run(
        self,
        model_name: str,
        samples: Optional[int] = None,
        experiment: Optional[str] = None,
        verbose: bool = True,
        project: Optional[str] = None,
        group: Optional[str] = None,
        job_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> wandb.Run:
        """Start a new wandb run with the specified configuration."""
        self.experiment = experiment
        
        if experiment:
            exp_config = ExperimentConfigManager.get_config(experiment)
            project = project or exp_config.project
            group = group or exp_config.group
            job_type = job_type or exp_config.job_type
            tags = tags or exp_config.tags
            config = config or exp_config.config_template(model_name)
        elif not all([project, group, job_type, config]):
            raise ValueError('No experiment defined and missing required configuration parameters')

        if verbose:
            self._print_config(experiment, project, group, job_type, tags, config)

        try:
            self.run = wandb.init(
                project=project,
                group=group,
                tags=tags,
                job_type=job_type,
                config=config
            )
            self.artifact_manager = ArtifactManager(self.run)
            return self.run
        except Exception as e:
            raise RuntimeError(f"Failed to initialize wandb run: {str(e)}")

    def _print_config(
        self,
        experiment: Optional[str],
        project: str,
        group: str,
        job_type: str,
        tags: List[str],
        config: Dict[str, Any]
    ) -> None:
        """Print the configuration details for the run."""
        print('Configuration:')
        print(f'experiment: {experiment}')
        print(f'project: {project}')
        print(f'group: {group}')
        print(f'job_type: {job_type}')
        print(f'tags: {tags}')
        for k, v in config.items():
            print(f"{k:{15}} : {v}")
        print()

    def stop_run(self) -> None:
        """Stop the current wandb run if one exists."""
        if self.run is not None:
            self.run.finish()

    def upload_model(self, model_path: str) -> Optional[wandb.Artifact]:
        """Upload a model artifact to wandb."""
        if not self.artifact_manager:
            raise RuntimeError("No active wandb run")
        return self.artifact_manager.upload_model(model_path, self.experiment)

    def download_model(self, model_id: str) -> str:
        """Download a specific version of the model artifact from wandb."""
        if not self.artifact_manager:
            raise RuntimeError("No active wandb run")
        return self.artifact_manager.download_model(model_id)

    def log_metrics(self, 
                   tracker: FunctionTracker, 
                   true_labels: pd.Series, 
                   predicted_labels: pd.Series
                   ) -> Dict[str, Any]:
        """Log metrics and tables to the wandb run dashboard."""
        if self.run is None:
            raise RuntimeError("No active wandb run")
            
        if self.results_df is None:
            raise RuntimeError("No results DataFrame available. Call log_evaluation first.")
            
        if not self.metrics:
            self.metrics = MetricsCalculator.calculate_metrics(true_labels, predicted_labels)

        efficiency_metrics = tracker.get_metrics()
        
        metrics = {
            "model": self.run.config.model_name,
            "eval_accuracy": self.metrics["accuracy"],
            "eval_macro_f1": self.metrics["macro_f1"],
            "eval_samples": len(self.results_df),
            "eval_C02eq": efficiency_metrics['Emissions (CO2eq)']['inference'],
            "eval_kWh": efficiency_metrics['Energy (kWh)']['inference'],
            "eval_runtime": efficiency_metrics['Timings (seconds)']['inference'],
            "eval_results": wandb.Table(dataframe=self.results_df),
            "eval_scores": wandb.Table(dataframe=self.metrics["category_df"]),
            "run_efficiency": wandb.Table(dataframe=efficiency_metrics),
        }
        
        self.run.log(metrics)
        return metrics

    def log_evaluation(self, results: pd.DataFrame, tracker: FunctionTracker) -> Dict[str, Any]:
        """Log both results artifact and metrics to the run."""
        if not self.artifact_manager:
            raise RuntimeError("No active wandb run")
            
        # Validate input DataFrame
        required_columns = ['label_true', 'label_pred']
        if not all(col in results.columns for col in required_columns):
            raise ValueError(f"Results DataFrame must contain columns: {required_columns}")
            
        self.results_df = results
        self.metrics = MetricsCalculator.calculate_metrics(
            true_labels=results['label_true'],
            predicted_labels=results['label_pred']
        )
        
        self.artifact_manager.log_results_artifact(results, self.metrics)
        return self.log_metrics(tracker=tracker)