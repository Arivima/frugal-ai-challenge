# utils.sampling.py
    # sample_balanced_subset

from datasets import DatasetDict, Dataset, concatenate_datasets
from collections import Counter

# Function to sample data while maintaining balance
def sample_balanced_subset(dataset, label_col= "label", N=None, seed=42, get_indexes=False):
    """
    Samples a balanced subset (from a Hugging Face Dataset or DatasetDict).
    Returns only a list of selected indexes if get_indexes is True, otherwise returns the subset.
    """
    def sample_from_Dataset(ds, label_col, N, seed):
        labels = ds.unique(label_col)
        num_classes = len(labels)
        class_counts = Counter(ds[label_col])
        min_class = min(class_counts.values())
        
        if N is None:
            samples_per_class = min_class
        else:
            samples_per_class = max(1, N // num_classes)
            if min_class < samples_per_class:
                samples_per_class = min_class

        subset = []

        for label in labels:
            # Filter dataset by label
            class_samples = ds.filter(lambda x: x[label_col] == label)
            class_samples = class_samples.shuffle(seed=seed)

            # Select samples
            selection_size = min(samples_per_class, len(class_samples))
            selected_samples = class_samples.select(range(selection_size))

            subset.append(selected_samples)

        return concatenate_datasets(subset) if subset else ds

    if dataset is None or dataset.num_rows == 0:
        return
    
    if isinstance(dataset, Dataset):
        return sample_from_Dataset(dataset, label_col, N, seed)
    
    if isinstance(dataset, DatasetDict):
        return DatasetDict({
            split: sample_from_Dataset(dataset[split], label_col, N, seed)
            for split in dataset.keys()
            })
