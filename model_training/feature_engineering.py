"""Feature selection and network-flow interaction features."""

import pandas as pd


RAW_FEATURES = [
    "spkts",
    "dpkts",
    "sload",
    "dload",
    "ct_state_ttl",
]


def extract_features(df):
    """Return the nine features used by the classifier."""
    eps = 1e-6
    features = pd.DataFrame(index=df.index)

    for column in RAW_FEATURES:
        features[column] = df[column]

    features["pkt_ratio"] = (df["spkts"] + eps) / (df["dpkts"] + eps)
    features["total_pkts"] = df["spkts"] + df["dpkts"]
    features["load_ratio"] = (df["sload"] + eps) / (df["dload"] + eps)
    features["total_load"] = df["sload"] + df["dload"]

    return features
