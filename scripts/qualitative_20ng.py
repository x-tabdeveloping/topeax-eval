import numpy as np
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from tqdm import trange
from turftopic import BERTopic, Top2Vec, Topeax

ds = fetch_20newsgroups(
    subset="all",
    remove=("headers", "footers", "quotes"),
)
corpus = ds.data
true_labels = ds.target
encoder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = np.load("embedding_cache/20news_all-MiniLM.npy")

model = Topeax(encoder=encoder, random_state=42)
doc_topic = model.fit_transform(corpus, embeddings=embeddings)

corr_m = np.zeros((len(ds.target_names), model.components_.shape[0]))
for i in trange(len(ds.target_names)):
    for j in range(model.components_.shape[0]):
        corr_m[i, j] = metrics.adjusted_mutual_info_score(
            true_labels == i, model.labels_ == j
        )
colorscale = [
    "#01014B",
    "#5D5DEF",
    "#B7B7FF",
    "#ffffff",
][::-1]
color_grid = [0.0, 0.4, 0.6, 1.0]
fig = px.imshow(
    corr_m.T,
    y=model.topic_names,
    x=ds.target_names,
    color_continuous_scale=list(zip(color_grid, colorscale)),
)
fig = fig.update_coloraxes(showscale=False)
fig = fig.update_layout(
    width=1000,
    height=600,
    font=dict(family="Merriweather", size=16, color="black"),
    margin=dict(t=0, b=0, l=0, r=0),
    yaxis={"side": "right"},
)
fig.show()
fig.write_image("figures/cluster_overlap_20news.png", scale=2)

fig = model.plot_density(light_mode=True)
fig = fig.update_layout(width=600, height=600)
fig = fig.for_each_annotation(
    lambda a: a.update(text=a.text.split("<br>")[0], yshift=0)
)
fig = fig.update_layout(
    template="plotly_white", paper_bgcolor="#DFDFFF", plot_bgcolor="#DFDFFF"
)
fig = fig.update_xaxes(showgrid=False, zeroline=False)
fig = fig.update_yaxes(showgrid=False, zeroline=False)
fig.show()
fig.write_image("figures/density_20news.png", scale=2)

top2vec = Top2Vec(encoder=encoder, random_state=42)
top2vec.fit(corpus, embeddings=embeddings)

print("------- Top2Vec ----------")
top2vec.print_topics()
proportion_outlier = (top2vec.labels_ == -1).sum() / len(corpus)
print("Percentage of outliers: ", 100 * proportion_outlier)

bertopic = BERTopic(encoder=encoder, random_state=42)
bertopic.fit(corpus, embeddings=embeddings)

print("------- BERTopic ----------")
bertopic.print_topics()
proportion_outlier = (bertopic.labels_ == -1).sum() / len(corpus)
print("Percentage of outliers: ", 100 * proportion_outlier)
