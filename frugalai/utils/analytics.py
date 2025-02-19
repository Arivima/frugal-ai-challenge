from collections import Counter
from datasets import DatasetDict, Dataset
import pandas as pd

# Count occurrences of each class label
def print_distribution(dataset: DatasetDict | Dataset | pd.Series | list, column: str = 'label'):
    """Prints the distribution of a specified column in a Hugging Face dataset, Pandas Series, or list."""
    
    def print_counter(category_counter : Counter):
        """Helper function to print the category distribution in a formatted table."""
        total = sum(category_counter.values())
        if total == 0:
            print("No data available.")
            return
        print(f"{'Category':<40}{'Count':<10}{'Percentage'}")
        print("-" * 60)
        for category, count in sorted(category_counter.items()):
            percentage = round((count / total) * 100, 2)
            print(f"{category:<40}{count:<10}{percentage}%")

    if isinstance(dataset, DatasetDict):
        for split in dataset.keys():
            print(f"\n🔹 Label distribution in '{split}' split (dataset.DatasetDict):")
            print_counter(Counter(dataset[split][column]))

    elif isinstance(dataset, Dataset):
        print("\n🔹 Label distribution (dataset.Dataset):")
        print_counter(Counter(dataset[column]))
    
    elif isinstance(dataset, pd.Series):
        print("\n🔹 Label distribution (pd.Series):")
        print_counter(Counter(dataset))

    elif isinstance(dataset, list):
        print("\n🔹 Label distribution (list):")
        print_counter(Counter(dataset))

    else : 
        print("❌ Unsupported dataset type. Provide a DatasetDict, Dataset, Pandas Series, or list.")


if __name__ == "__main__":
    from datasets import load_dataset

    ds = load_dataset("QuotaClimat/frugalaichallenge-text-train")
    print_distribution(ds, 'label')
