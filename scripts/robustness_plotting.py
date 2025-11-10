import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots


def load_results(file_path):
    entries = []
    with file_path.open() as in_file:
        for line in in_file:
            entry = json.loads(line.strip())
            entries.append(entry)
    df = pd.DataFrame.from_records(entries)
    df["coherence"] = np.sqrt(df["c_ex"] * df["c_in"])
    df["topic_quality"] = np.sqrt(df["diversity"] * df["coherence"])
    return df


COLOR_MAPPING = {
    "Topeax": "#5D5DEF",
    "Top2Vec": "#8AD0B5",
    "BERTopic": "#CF5286",
}
METRICS = [
    ("fowlkes_mallows_score", "FMI"),
    ("topic_quality", "Interpretability"),
    ("n_components", "No. Topics"),
]
ROBUSTNESS_DIR = Path("results/robustness")
FILES = [
    ("subsampling.jsonl", "sample_size", "Sample Size"),
    ("perplexity.jsonl", "perplexity", "Perplexity"),
]

for (
    file_name,
    x,
    x_name,
) in FILES:
    df = load_results(ROBUSTNESS_DIR.joinpath(file_name))
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.08)
    for col, (y, y_name) in enumerate(METRICS):
        subfig = px.line(
            df,
            x=x,
            y=y,
            color="model",
            color_discrete_map=COLOR_MAPPING,
        )
        subfig.update_traces(showlegend=col == 0)
        for trace in subfig.data:
            fig.add_trace(trace, col=col + 1, row=1)
        fig = fig.update_yaxes(title=y_name, col=col + 1, row=1)
    fig = fig.add_scatter(
        x=np.linspace(2, df[x].max(), 5),
        y=np.full(5, 20),
        line=dict(dash="dash", color="black", width=2),
        name="Gold Labels",
        mode="lines",
        col=3,
        row=1,
    )
    fig = fig.update_xaxes(title=x_name)
    fig = fig.update_traces(line=dict(width=4))
    fig = fig.update_layout(
        width=1400,
        height=400,
        template="plotly_white",
        margin=dict(b=0, t=20, l=0, r=0),
        font=dict(size=18, family="Merriweather", color="black"),
    )
    fig.show()
    fig.write_image(f"figures/robustness_{x}.png", scale=2)
