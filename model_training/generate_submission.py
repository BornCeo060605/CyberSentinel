"""Generate the final prediction CSV from the trained model."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from data_loader import load_data
from feature_engineering import extract_features
from train_model import MODEL_PATH, THRESHOLD


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_PATH = PROJECT_ROOT / "submission.csv"


def create_submission(output_path=SUBMISSION_PATH):
    _, validation_df = load_data()
    model = joblib.load(MODEL_PATH)

    X_validation = extract_features(validation_df)
    attack_probabilities = model.predict_proba(X_validation)[:, 1]

    predictions = np.where(
        attack_probabilities >= THRESHOLD,
        "attack",
        "normal",
    )
    confidences = np.where(
        predictions == "attack",
        attack_probabilities,
        1.0 - attack_probabilities,
    )

    submission = pd.DataFrame({
        "sample_id": [f"CSHT_{i:04d}" for i in range(len(validation_df))],
        "predicted_class": predictions,
        "confidence": np.round(confidences, 4),
    })

    submission.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path}")
    print("\nPrediction counts:")
    print(submission["predicted_class"].value_counts())

    return submission


if __name__ == "__main__":
    create_submission()
