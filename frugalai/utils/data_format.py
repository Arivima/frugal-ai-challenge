from datasets import load_dataset, DatasetDict

system_msg = """You are a climate statement classifier. 
Your task is to categorize statements by identifying which type of climate narrative they represent.
"""

prompt_template = """Classify the following statement into exactly one category (0-7).
Respond with ONLY the category number.

### Categories:
0 - Not relevant: No climate-related claims or doesn't fit other categories
1 - Denial: Claims climate change is not happening
2 - Attribution denial: Claims human activity is not causing climate change
3 - Impact minimization: Claims climate change impacts are minimal or beneficial
4 - Solution opposition: Claims solutions to climate change are harmful
5 - Science skepticism: Challenges climate science validity or methods
6 - Actor criticism: Attacks credibility of climate scientists or activists
7 - Fossil fuel promotion: Asserts importance of fossil fuels

### Statement to classify: "{}"
### Answer:
"""

def load_frugalai_dataset(
    select_splits: list[str] | None = None,  # e.g. ['train', 'test', 'validation']; if None, return all splits
    sample: int | float | None = None         # selects the first `sample` samples of each split in the dataset
    ):
    """Load dataset and format it according to the model's template"""
    def encode_target(examples):
        unique_labels = sorted(set(examples["label"]))
        label2id = {label: i for i, label in enumerate(unique_labels)}
        encoded_label = [label2id[e] for e in examples["label"]]
        return {"label" : encoded_label}
    
    dataset = load_dataset("QuotaClimat/frugalaichallenge-text-train")
    
    # Create train/val split
    train_val_split = dataset['train'].train_test_split(test_size=0.2, seed=42)
    dataset = DatasetDict({
        'train': train_val_split['train'],
        'validation': train_val_split['test'],
        'test': dataset['test']
    })
    # select columns / encode target
    for split in dataset:
        dataset[split] = dataset[split].select_columns(['quote', 'label'])
        dataset[split] = dataset[split].map(encode_target, batched=True)

    # Apply sample selection if provided.
    if sample is not None:
        for split in dataset:
            valid_sample = None
            # If sample is an integer and within bounds, use it directly.
            if isinstance(sample, int) and 0 < sample <= len(dataset[split]):
                valid_sample = sample
            # If sample is a float between 0 and 1, treat it as a fraction.
            elif isinstance(sample, float) and 0.0 < sample < 1.0:
                valid_sample = int(sample * len(dataset[split]))
            
            if valid_sample is not None:
                dataset[split] = dataset[split].select(range(valid_sample))
    
    # If specific splits are requested, filter and return them.
    if select_splits:
        filtered_dataset = {split: dataset[split] for split in select_splits if split in dataset}
        return filtered_dataset
    
    # otherwise, returns the whole dataset
    return dataset


def create_formatting_function(
        tokenizer,
        force_prompt=False,  # if True forces the use of generic prompt
        system_msg=system_msg,  # can be overriden to experiment
        prompt_template=prompt_template  # can be overriden to experiment
    ):
    """Creates a function to format prompts for the model.
    
    Args:
        tokenizer: The tokenizer to use for formatting
        force_prompt: If True, forces use of generic prompt format
        system_msg: System message to use in prompts
        prompt_template: Template for formatting user prompts
        
    Returns:
        Function that formats examples into prompts
    """
    def _get_quotes(examples):
        """Extract quotes from examples input."""
        return examples if isinstance(examples, list) else examples["text"]
    
    def _format_with_chat_template(user_msgs, tokenizer):
        """Format messages using tokenizer's chat template."""
        messages = [
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": msg}
            ] for msg in user_msgs
        ]
        return [
            tokenizer.apply_chat_template(
                msg,
                tokenize=False,  # Trainer will tokenize later
                add_generation_prompt=True  # adds assistant token
            ) for msg in messages
        ]

    def _format_generic(user_msgs):
        """Format messages using generic template."""
        return [
            f"### System:\n{system_msg}\n\n### User:\n{msg}\n\n### Assistant:"
            for msg in user_msgs
        ]

    def _should_use_chat_template(tokenizer, force_prompt):
        """Determine if chat template should be used."""
        if force_prompt:
            return False
        return (hasattr(tokenizer, "chat_template") and 
                hasattr(tokenizer, "apply_chat_template"))

    def formatting_prompts_func(examples):
        """Format examples into prompts.
        
        Args:
            examples: List of quotes or dataset with 'text' column
            
        Returns:
            Formatted prompts in same format as input
        """
        quotes = _get_quotes(examples)
        user_msgs = [prompt_template.format(quote) for quote in quotes]
        
        use_chat = _should_use_chat_template(tokenizer, force_prompt)
        formatted_quotes = (_format_with_chat_template(user_msgs, tokenizer) 
                          if use_chat else _format_generic(user_msgs))

        return formatted_quotes if isinstance(examples, list) else {"text": formatted_quotes}
    
    return formatting_prompts_func
