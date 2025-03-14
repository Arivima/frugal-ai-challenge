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

def load_frugalai_dataset():
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
    for split in dataset:
        dataset[split] = dataset[split].select_columns(['quote', 'label'])
        dataset[split] = dataset[split].map(encode_target, batched=True)

    return dataset


def create_formatting_function(
        tokenizer, 
        force_prompt = False, # if True forces the use of generic prompt
        system_msg=system_msg, #can be overriden to experiment
        prompt_template=prompt_template #can be overriden to experiment
    ):

    # Check if the model has a chat template
    has_chat_template = hasattr(tokenizer, "chat_template") if force_prompt == False else None
    has_apply_chat_template = hasattr(tokenizer, "apply_chat_template") if force_prompt == False else None

    def formatting_prompts_func(examples):
        # examples can be a list of quotes or a dataset with a column `quote``
        if isinstance(examples, list):
            quotes = examples
        else:
            quotes = examples["quote"]

        formatted_quotes = []
        user_msgs = [prompt_template.format(quote) for quote in quotes]

        if has_chat_template and has_apply_chat_template:
            all_messages = [
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ] for user_msg in user_msgs
            ]
            formatted_quotes = [
                tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, # Trainer will tokenize later
                    add_generation_prompt=True # adds assistant token
                ) for messages in all_messages
            ]

        else:
            # generic format if no chat template available
            # this can also be forced with arg `force_prompt`
            formatted_quotes = [
                f"### System:\n{system_msg}\n\n### User:\n{user_msg}\n\n### Assistant:"
                for user_msg in user_msgs
            ]
        
        # output in the same format as input
        if isinstance(examples, list):
            return formatted_quotes
        else:
            return {"text": formatted_quotes}
    
    return formatting_prompts_func


