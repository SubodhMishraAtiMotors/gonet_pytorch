#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    predictions_path = Path(args.predictions)

    if args.output is None:
        output_path = predictions_path.parent / "threshold_sweep.csv"
    else:
        output_path = Path(args.output)

    df = pd.read_csv(predictions_path)

    y_true = df["label"].astype(int).values
    y_prob = df["prob_traversable"].astype(float).values

    rows = []

    for threshold in np.arange(0.05, 0.96, 0.05):
        y_pred = (y_prob >= threshold).astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        fpr = fp / max(1, fp + tn)
        fnr = fn / max(1, fn + tp)

        rows.append({
            "threshold": round(float(threshold), 3),
            "accuracy": acc,
            "precision_go": precision,
            "recall_go": recall,
            "f1_go": f1,
            "tn_no_go_correct": int(tn),
            "fp_no_go_predicted_go": int(fp),
            "fn_go_predicted_no_go": int(fn),
            "tp_go_correct": int(tp),
            "false_positive_rate_no_go_as_go": fpr,
            "false_negative_rate_go_as_no_go": fnr,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)

    print(out_df.to_string(index=False))
    print()
    print(f"Saved threshold sweep to: {output_path}")


if __name__ == "__main__":
    main()