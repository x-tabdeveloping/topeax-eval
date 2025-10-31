import argparse
import json
from itertools import chain, combinations
from pathlib import Path

import gensim.downloader as api
import mteb
import numpy as np
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
    return float(len(unique_words) / total_words)


def word_embedding_coherence(keywords, wv):
    arrays = []
    for index, topic in enumerate(keywords):
        if len(topic) > 0:
            local_simi = []
            for word1, word2 in combinations(topic, 2):
                if word1 in wv.index_to_key and word2 in wv.index_to_key:
                    local_simi.append(wv.similarity(word1, word2))
            arrays.append(np.nanmean(local_simi))
    return float(np.nanmean(arrays))


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


def load_cache(out_path):
    cache_entries = []
    with out_path.open() as cache_file:
        for line in cache_file:
            entry = json.loads(line.strip())
            cache_entry = (entry["task"], entry["model"])
            cache_entries.append(cache_entry)
    return set(cache_entries)


def main(encoder_name: str = "all-MiniLM-L6-v2"):
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    encoder_path_name = encoder_name.replace("/", "__")
    out_path = out_dir.joinpath(f"{encoder_path_name}.jsonl")
    if out_path.is_file():
        cache = load_cache(out_path)
    else:
        cache = set()
        # Create file if doesn't exist
        with out_path.open("w"):
            pass
    print("Loading external word embeddings")
    ex_wv = api.load("word2vec-google-news-300")
    print("Loading benchmark")
    benchmark = mteb.get_benchmark("MTEB(eng, v2)")
    tasks = mteb.filter_tasks(benchmark.tasks, task_types=["Clustering"])
    for task in tasks:
        if all(
            [(task.metadata.name, model_name) in cache for model_name in topic_models]
        ):
            print("All models already completed, skipping.")
            continue
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
        glove = GloVe(vector_size=50)
        tokenized_corpus = [tokenizer(text) for text in corpus]
        glove.train(tokenized_corpus)
        in_wv = glove.wv
        encoder = SentenceTransformer(encoder_name)
        print("Encoding task corpus.")
        embeddings = encoder.encode(corpus, show_progress_bar=True)
        for model_name in topic_models:
            if (task.metadata.name, model_name) in cache:
                print(f"{model_name} already done, skipping.")
                continue
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
                **clust_scores,
                **topic_scores,
            }
            print("Results: ", res)
            res["keywords"] = keywords
            with out_path.open("a") as out_file:
                out_file.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Evaluate clustering.")
    parser.add_argument("embedding_model")
    args = parser.parse_args()
    encoder = args.embedding_model
    main(encoder)
