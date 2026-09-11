"""Feature ranking using ANOVA F-test and mutual information."""

import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif

from data_loader import load_data


def run_feature_audit(train_df):
    y_binary = (train_df["label"] != "normal").astype(int)
    feature_columns = [c for c in train_df.columns if c != "label"]

    f_scores, p_values = f_classif(train_df[feature_columns], y_binary)
    mi_scores = mutual_info_classif(
        train_df[feature_columns], y_binary, random_state=42
    )

    ranking = pd.DataFrame({
        "Feature": feature_columns,
        "F-Statistic": f_scores,
        "p-value": p_values,
        "Mutual-Info": mi_scores,
    }).sort_values("F-Statistic", ascending=False)

    print(ranking.to_string(index=False))
    return ranking


if __name__ == "__main__":
    train_df, _ = load_data()
    run_feature_audit(train_df)
