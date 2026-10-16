"""Report and check active-parameter budgets without allocating model weights."""

import argparse
import json

from moelab.config import load_model_config
from moelab.model.budget import compare_budgets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense", default="configs/model/compare-dense.yaml")
    parser.add_argument("--moe", default="configs/model/compare-moe.yaml")
    args = parser.parse_args()
    report = compare_budgets(load_model_config(args.dense), load_model_config(args.moe))
    print(json.dumps(report, indent=2))
    if not report["within_0_1_percent"]:
        parser.exit(1, "Active-parameter gap exceeds the declared 0.1% tolerance.\n")
