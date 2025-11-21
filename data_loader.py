import pandas as pd
from datasets import load_dataset

def load_data():
    ds = load_dataset("JosefGoldstein/aimlessinnovations_customer_sentiment")

    # We ll need only train data for this experiment
    data = ds["train"]

    # Convert to pandas DataFrame for easier manipulation
    df = pd.DataFrame(data)

    # Get 2 samples per label
    samples = df.groupby('label').head(2)
    return samples

if __name__ == "__main__":
    samples = load_data()

    print(samples.columns)

    # Display the results
    for label, group in samples.groupby('label'):
        print(f"\nLabel: {label}")
        for _, row in group.iterrows():
            print(f"Text: {row['text']}")
            print(f"Sentiment: {row['label_text']}\n")
