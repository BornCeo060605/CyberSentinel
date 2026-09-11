"""Train and calibrate the network intrusion classifier."""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from data_loader import load_data
from feature_engineering import extract_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "calibrated_model.joblib"
THRESHOLD = 0.74


def build_model():
    base_estimator = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.03,
        max_leaf_nodes=15,
        min_samples_leaf=25,
        l2_regularization=3.0,
        random_state=42,
    )

    return CalibratedClassifierCV(
        estimator=base_estimator,
        method="sigmoid",
        cv=5,
    )


def train_and_evaluate():
    train_df, validation_df = load_data()

    y_train = (train_df["label"] != "normal").astype(int)
    y_validation = (validation_df["label"] != "normal").astype(int)

    X_train = extract_features(train_df)
    X_validation = extract_features(validation_df)

    model = build_model()
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_validation)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    print(f"Validation accuracy: {accuracy_score(y_validation, predictions):.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_validation,
            predictions,
            target_names=["normal", "attack"],
            digits=4,
        )
    )
    print("Confusion matrix:")
    print(confusion_matrix(y_validation, predictions))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    return model


if __name__ == "__main__":
    train_and_evaluate()
