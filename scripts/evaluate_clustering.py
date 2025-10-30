import argparse
from itertools import chain, combinations
from pathlib import Path

import gensim.downloader as api
import mteb
import numpy as np
import pandas as pd
from glovpy import GloVe
from sentence_transformers import SentenceTransformer
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from turftopic import BERTopic, Top2Vec, Topeax

topic_models = {
    "Topeax": lambda encoder: Topeax(
        encoder=encoder, vectorizer=CountVectorizer(), random_state=42
    ),
    "BERTopic": lambda encoder: BERTopic(
        encoder=encoder, vectorizer=CountVectorizer(), random_state=42
    ),
    "Top2Vec": lambda encoder: Top2Vec(
        encoder=encoder, vectorizer=CountVectorizer(), random_state=42
    ),
}


def diversity(keywords: list[list[str]]) -> float:
    all_words = list(chain.from_iterable(keywords))
    unique_words = set(all_words)
    total_words = len(all_words)
    return len(unique_words) / total_words


def word_embedding_coherence(keywords, wv):
    arrays = []
    for index, topic in enumerate(keywords):
        if len(topic) > 0:
            local_simi = []
            for word1, word2 in combinations(topic, 2):
                if word1 in wv.index_to_key and word2 in wv.index_to_key:
                    local_simi.append(wv.similarity(word1, word2))
            arrays.append(np.nanmean(local_simi))
    return np.nanmean(arrays)


def evaluate_clustering(true_labels, pred_labels) -> dict[str, float]:
    res = {}
    for metric in [
        metrics.fowlkes_mallows_score,
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.adjusted_mutual_info_score,
    ]:
        res[metric.__name__] = metric(true_labels, pred_labels)
    return res


def get_keywords(model) -> list[list[str]]:
    """Get top words and ignore outlier topic."""
    n_topics = model.components_.shape[0]
    try:
        classes = model.classes_
    except AttributeError:
        classes = list(range(n_topics))
    res = []
    for topic_id, words in zip(classes, model.get_top_words()):
        if topic_id != -1:
            res.append(words)
    return res


def evaluate_topic_quality(keywords, ex_wv, in_wv) -> dict[str, float]:
    res = {
        "diversity": diversity(keywords),
        "c_in": word_embedding_coherence(keywords, in_wv),
        "c_ex": word_embedding_coherence(keywords, ex_wv),
    }
    return res


def main(encoder_name: str = "all-MiniLM-L6-v2"):
    print("Loading external word embeddings")
    ex_wv = api.load("word2vec-google-news-300")
    print("Loading benchmark")
    benchmark = mteb.get_benchmark("MTEB(eng, v2)")
    tasks = mteb.filter_tasks(benchmark.tasks, task_types=["Clustering"])
    results = []
    for task in tasks:
        print(f"Loading data for task {task.metadata.name}")
        task.load_data()
        ds = task.dataset["test"]
        corpus = list(ds["sentences"])
        if isinstance(ds["labels"][0], list):
            true_labels = [label[0] for label in ds["labels"]]
        else:
            true_labels = list(ds["labels"])
        print("Training internal word embeddings using GloVe...")
        tokenizer = CountVectorizer().build_analyzer()
        in_wv = GloVe(vector_size=50)
        tokenized_corpus = [tokenizer(text) for text in corpus]
        in_wv.train(tokenized_corpus)
        encoder = SentenceTransformer(encoder_name)
        print("Encoding task corpus.")
        embeddings = encoder.encode(corpus, show_progress_bar=True)
        for model_name in topic_models:
            print(f"Running {model_name}.")
            model = topic_models[model_name](encoder)
            model.fit(corpus, embeddings=embeddings)
            keywords = get_keywords(model)
            print("Evaluating model.")
            clust_scores = evaluate_clustering(true_labels, model.labels_)
            topic_scores = evaluate_topic_quality(keywords, ex_wv, in_wv)
            res = {
                "encoder": encoder_name,
                "task": task.metadata.name,
                "model": model_name,
                "n_components": model.components_.shape[0],
                "true_n": len(set(true_labels)),
                "keywords": keywords,
                **clust_scores,
                **topic_scores,
            }
            results.append(res)
    res_df = pd.DataFrame.from_records(results)
    out_name = encoder_name.replace("/", "__")
    Path("results").mkdir(exist_ok=True)
    res_df.to_csv(f"results/mteb_clustering_{out_name}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Evaluate clustering.")
    parser.add_argument("embedding_model")
    args = parser.parse_args()
    encoder = args.embedding_model
    main(encoder)
