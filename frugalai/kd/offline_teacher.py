from typing import Dict, List, Optional, Union
import torch
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from tqdm import tqdm
import numpy as np
from unsloth import FastLanguageModel

from frugalai.utils.data_format import load_frugalai_dataset, create_formatting_function

# Custom system and prompt messages for knowledge distillation
KD_SYSTEM_MSG = """You are a climate statement classifier and teacher. 
Your task is to:
1. Categorize statements by identifying which type of climate narrative they represent
2. Provide evidence-based explanations for your classification, citing specific elements from the text

Be precise and concise in your explanations. When faced with ambiguous cases, explain your reasoning for choosing one category over others."""

KD_PROMPT_TEMPLATE = """### Categories:
0 - Not relevant: No climate-related claims or doesn't fit other categories
1 - Denial: Claims climate change is not happening
2 - Attribution denial: Claims human activity is not causing climate change
3 - Impact minimization: Claims climate change impacts are minimal or beneficial
4 - Solution opposition: Claims solutions to climate change are harmful
5 - Science skepticism: Challenges climate science validity or methods
6 - Actor criticism: Attacks credibility of climate scientists or activists
7 - Fossil fuel promotion: Asserts importance of fossil fuels

### Statement to classify: "{text}"

Classify the statement into exactly one category (0-7) and explain your reasoning.
Respond in the following JSON format:
{
    "text": "the original statement",
    "category": "the numeric category (0-7)",
    "explanation": "your detailed reasoning with specific evidence from the text"
}

### Example Response:
{
    "text": "Climate change is a hoax invented by scientists",
    "category": 1,
    "explanation": "This statement directly denies climate change by claiming it's a hoax, which fits category 1 - Denial. The mention of 'invented by scientists' also shows distrust in scientific institutions."
}

### Answer:
"""

class OfflineTeacher:
    def __init__(
        self,
        model_name: str = "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """Initialize the offline teacher model.
        
        Args:
            model_name: Name of the model to use
            device: Device to run the model on
            max_length: Maximum length of generated text
            temperature: Temperature for text generation
            top_p: Top-p sampling parameter
        """
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
                
        # Load model and tokenizer using Unsloth
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_length,
            dtype=None,  # Will use the model's default dtype
            load_in_4bit=True,
            rope_scaling = None  # Disable rope_scaling to avoid the dimension mismatch
            # Unsloth will automatically handle device mapping
        )
        
        # Create formatting function
        self.formatting_func = create_formatting_function(
            self.tokenizer,
            system_msg=KD_SYSTEM_MSG,
            prompt_template=KD_PROMPT_TEMPLATE
        )
    
    def generate_explanation(self, text: str) -> str:
        """Generate classification and explanation for a single text.
        
        Args:
            text: Input text to classify
            
        Returns:
            Generated explanation
        """
        # Format the prompt
        formatted_prompt = self.formatting_func([text])[0]
        
        # Tokenize input
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and extract the explanation
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the assistant's response
        response = response.split("### Answer:")[-1].strip()
        
        return response
    
    def generate_dataset(
        self,
        dataset: Optional[DatasetDict] = None,
        sample_size: Optional[Union[int, float]] = None,
        batch_size: int = 8
    ) -> Dataset:
        """Generate synthetic dataset for knowledge distillation.
        
        Args:
            dataset: Input dataset to generate explanations for
            sample_size: Number or fraction of samples to use
            batch_size: Batch size for processing
            
        Returns:
            Dataset containing original texts and generated explanations
        """
        # Load dataset if not provided
        if dataset is None:
            dataset = load_frugalai_dataset(sample=sample_size)
        
        # Process each split
        synthetic_data = []
        
        for split in ["train", "validation"]:
            if split not in dataset:
                continue
                
            print(f"Processing {split} split...")
            for i in tqdm(range(0, len(dataset[split]), batch_size)):
                batch = dataset[split][i:i + batch_size]
                
                # Generate explanations for the batch
                for text in batch["text"]:
                    explanation = self.generate_explanation(text)
                    
                    # Parse the explanation to extract category and reasoning
                    try:
                        category = int(explanation.split("Category:")[1].split("\n")[0].strip())
                        reasoning = explanation.split("Explanation:")[1].strip()
                    except:
                        category = -1
                        reasoning = explanation
                    
                    synthetic_data.append({
                        "text": text,
                        "category": category,
                        "explanation": reasoning,
                        "split": split
                    })
        
        # Convert to dataset
        synthetic_dataset = Dataset.from_list(synthetic_data)
        
        return synthetic_dataset
