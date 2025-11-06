import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

results_path = Path("results/robustness/perplexity.jsonl")
result_files = results_path.glob("*.jsonl")
entries = []
with results_path.open() as in_file:
    for line in in_file:
        entry = json.loads(line.strip())
        entries.append(entry)
df = pd.DataFrame.from_records(entries)
df["coherence"] = np.sqrt(df["c_ex"] * df["c_in"])
df["topic_quality"] = np.sqrt(df["diversity"] * df["coherence"])
df["cluster_topic_quality"] = np.sqrt(df["topic_quality"] * df["fowlkes_mallows_score"])

COLOR_MAPPING = {
    "Topeax": "#5D5DEF",
    "Top2Vec": "#8AD0B5",
    "BERTopic": "#CF5286",
}

fig = px.line(
    df,
    x="perplexity",
    y="n_components",
    color="model",
    color_discrete_map=COLOR_MAPPING,
)
fig = fig.update_layout(
    width=500,
    height=400,
    template="plotly_white",
    margin=dict(b=0, t=20, l=0, r=0),
    font=dict(size=14, family="Merriweather"),
)
fig = fig.update_traces(line=dict(width=2))
fig = fig.update_xaxes(title="Perplexity")
fig = fig.update_yaxes(title="No. Topics")
fig = fig.add_scatter(
    x=np.linspace(2, 100, 5),
    y=np.full(5, 20),
    line=dict(dash="dash", color="black", width=2),
    name="Gold Labels",
    mode="lines",
)
fig.show()

fig = px.line(
    df,
    x="perplexity",
    y="fowlkes_mallows_score",
    color="model",
    color_discrete_map=COLOR_MAPPING,
)
fig = fig.update_traces(line=dict(width=2))
fig = fig.update_xaxes(title="Perplexity")
fig = fig.update_yaxes(title="FMI")
fig = fig.update_layout(
    width=500,
    height=400,
    template="plotly_white",
    margin=dict(b=0, t=20, l=0, r=0),
    font=dict(size=14, family="Merriweather"),
)
fig.show()


fig = px.line(
    df,
    x="perplexity",
    y="topic_quality",
    color="model",
    color_discrete_map=COLOR_MAPPING,
)
fig = fig.update_traces(line=dict(width=2))
fig = fig.update_xaxes(title="Perplexity")
fig = fig.update_yaxes(title="Topic Quality")
fig = fig.update_layout(
    width=500,
    height=400,
    template="plotly_white",
    margin=dict(b=0, t=20, l=0, r=0),
    font=dict(size=14, family="Merriweather"),
)
fig.show()
