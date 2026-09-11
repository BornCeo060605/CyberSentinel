"""Load and inspect the NIDS datasets."""

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "train.csv"
VALIDATION_PATH = PROJECT_ROOT / "data" / "validation.csv"


def load_data(train_path=TRAIN_PATH, validation_path=VALIDATION_PATH):
    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)
    return train_df, validation_df


def inspect_data(train_df, validation_df):
    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {validation_df.shape}")
    print(f"Missing values in train: {train_df.isnull().sum().sum()}")
    print(f"Missing values in validation: {validation_df.isnull().sum().sum()}")


if __name__ == "__main__":
    train_df, validation_df = load_data()
    inspect_data(train_df, validation_df)
