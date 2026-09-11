"""Evaluate a submission against the validation labels."""

from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_PATH = PROJECT_ROOT / "data" / "validation.csv"
SUBMISSION_PATH = PROJECT_ROOT / "submission.csv"


def evaluate():
    validation = pd.read_csv(VALIDATION_PATH)
    submission = pd.read_csv(SUBMISSION_PATH)

    y_true = (validation["label"] != "normal").astype(int)
    y_pred = (submission["predicted_class"] == "attack").astype(int)

    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["normal", "attack"],
            digits=4,
        )
    )


if __name__ == "__main__":
    evaluate()
