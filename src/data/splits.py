from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split

def simple_split(df: pd.DataFrame, target_cols, test_size=0.2, random_state=42):
    X = df.drop(columns=target_cols)
    y = df[target_cols]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
