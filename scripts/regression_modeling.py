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
df["coherence"] = np.sqrt(df["c_ex"] * df["c_in"])
df["topic_quality"] = np.sqrt(df["diversity"] * df["coherence"])
df["cluster_topic_quality"] = np.sqrt(df["topic_quality"] * df["fowlkes_mallows_score"])
df["n_diff"] = df["n_components"] - df["true_n"]
df["n_diff_percent"] = -100 * ((df["true_n"] - df["n_components"]) / df["true_n"])
df["model"] = pd.Categorical(
    df["model"], categories=["Topeax", "Top2Vec", "BERTopic"], ordered=True
)

reg_data = df[
    [
        "model",
        "topic_quality",
        "cluster_topic_quality",
        "encoder",
        "task",
        "fowlkes_mallows_score",
    ]
]
reg_data["groups"] = reg_data["encoder"].astype(str) + reg_data["task"]
rand_eff_mod = smf.ols(
    formula="fowlkes_mallows_score ~ C(model)",
    data=reg_data,
    groups=reg_data["groups"],
    re_formula="~task",
)
res = rand_eff_mod.fit()

print(res.summary().as_csv())
