import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import bootstrap

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
df["abs_diff_percent"] = np.abs(
    100 * ((df["true_n"] - df["n_components"]) / df["true_n"])
)
df["model"] = pd.Categorical(
    df["model"], categories=["Topeax", "Top2Vec", "BERTopic"], ordered=True
)

mape = df.groupby("model")["abs_diff_percent"].agg(["mean", "std"])
print(mape)


def bootstrap_interval(data) -> tuple[float, float, float]:
    res = bootstrap([data], np.mean)
    return (
        np.mean(data),
        res.confidence_interval.low,
        res.confidence_interval.high,
    )


df.groupby("model")[["c_in", "c_ex", "diversity", "topic_quality"]].agg(
    lambda d: f"{d.mean():.2f}±{d.std():.2f}"
)


def bootstrap_scatter(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    groups: str,
    color_mapping: Optional[dict[str, str]] = None,
) -> go.Figure:
    fig = go.Figure()
    for group_name, group_data in data.groupby(groups):
        x, x_low, x_high = bootstrap_interval(group_data[x_col])
        y, y_low, y_high = bootstrap_interval(group_data[y_col])
        marker = dict()
        if color_mapping is not None and group_name in color_mapping:
            marker["color"] = color_mapping[group_name]
            marker["line"] = dict(color="black", width=3)
        fig.add_trace(
            go.Scatter(
                name=group_name,
                text=[group_name],
                x=[x],
                y=[y],
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[x_high - x],
                    arrayminus=[x - x_low],
                    thickness=2.2,
                    width=4,
                    color="black",
                ),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[y_high - y],
                    arrayminus=[y - y_low],
                    thickness=2.2,
                    width=4,
                    color="black",
                ),
                showlegend=True,
                mode="markers",
                marker=marker,
            )
        )
    fig = fig.update_layout(template="plotly_white")
    fig = fig.update_traces(
        marker=dict(
            size=17,
            line=dict(
                color="black",
                width=2,
            ),
        )
    )
    return fig


color_mapping = {
    "Topeax": "#5D5DEF",
    "Top2Vec": "#8AD0B5",
    "BERTopic": "#CF5286",
}
fig = bootstrap_scatter(
    df,
    "fowlkes_mallows_score",
    "topic_quality",
    groups="model",
    color_mapping=color_mapping,
)
fig = fig.update_layout(
    width=600,
    height=350,
    template="plotly_white",
    font=dict(size=16, family="Merriweather", color="black"),
    xaxis_title="FMI",
    yaxis_title="Interpretablity",
    margin=dict(t=0, b=0, r=0, l=0),
)
fig = fig.add_scatter(
    x=df["fowlkes_mallows_score"],
    y=df["topic_quality"],
    marker=dict(
        color=[color_mapping[str(model)] for model in df["model"]],
        size=10,
        line=dict(color="black", width=2),
    ),
    opacity=1.0,
    showlegend=False,
    mode="markers",
)
fig.data = fig.data[::-1]
fig.show()

fig = px.box(
    df,
    x="model",
    y="abs_diff_percent",
    color="model",
    color_discrete_map=color_mapping,
    points="all",
)
fig = fig.update_traces(marker=dict(line=dict(color="black", width=2), size=10))
fig = fig.update_layout(
    width=400,
    height=350,
    template="plotly_white",
    font=dict(size=16, family="Merriweather", color="black"),
    yaxis_title="Percentage Error in N topics",
    xaxis_title=None,
    margin=dict(t=0, b=30, r=0, l=0),
)
fig = fig.update_traces(showlegend=False)
fig.show()


fig = px.box(
    df,
    y="model",
    x="cluster_topic_quality",
    color="model",
    color_discrete_map=color_mapping,
    points="all",
)
fig = fig.update_traces(marker=dict(line=dict(color="black", width=2), size=10))
fig = fig.update_layout(
    width=400,
    height=500,
    template="plotly_white",
    font=dict(size=16, family="Merriweather", color="black"),
    xaxis_title="Aggregate",
    yaxis_title=None,
    margin=dict(t=0, b=30, r=0, l=0),
)
fig = fig.update_traces(showlegend=False)
fig.show()
