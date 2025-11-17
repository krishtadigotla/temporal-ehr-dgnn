import pandas as pd

def load_dataset():
    path = "data/full_cohort_data.csv"
    df = pd.read_csv(path)

    # Convert column names to lowercase for consistency
    df.columns = [c.lower() for c in df.columns]

    return df

if __name__ == "__main__":
    df = load_dataset()
    print("Shape:", df.shape)
    print(df.head())
