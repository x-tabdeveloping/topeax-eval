"""Script for running regression model to predict FMI from model type"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

results_path = Path("results/")
result_files = results_path.glob("*.jsonl")
entries = []
for file_path in result_files:
    if file_path.stem.startswith("_"):
        # Skipping over files marked as not important
        continue
    with file_path.open() as in_file:
        for line in in_file:
            entry = json.loads(line.strip())
            entries.append(entry)
df = pd.DataFrame.from_records(entries)
# Making sure that Topeax is the intercept
df["model"] = pd.Categorical(
    df["model"], categories=["Topeax", "Top2Vec", "BERTopic"], ordered=True
)

reg_data = df[
    [
        "model",
        "encoder",
        "task",
        "fowlkes_mallows_score",
    ]
]
# Grouping by encoder and task
reg_data["groups"] = reg_data["encoder"].astype(str) + reg_data["task"]
# Regression model predicting FMI from model with random intercepts for encoder-task pairs and random effect for task
rand_eff_mod = smf.ols(
    formula="fowlkes_mallows_score ~ C(model)",
    data=reg_data,
    groups=reg_data["groups"],
    re_formula="~task",
)
res = rand_eff_mod.fit()

# Printing regression coefficients as csv
print(res.summary().as_csv())
