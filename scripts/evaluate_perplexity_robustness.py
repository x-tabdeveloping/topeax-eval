import json
from pathlib import Path

import gensim.downloader as api
import numpy as np
from evaluate_clustering import (evaluate_clustering, evaluate_topic_quality,
                                 get_keywords)
from glovpy import GloVe
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from turftopic import BERTopic, Top2Vec, Topeax
from umap import UMAP


def create_umap(perplexity: int):
    """Convenience function for initializing UMAP with a given perplexity"""
    dimensionality_reduction = UMAP(
        n_neighbors=perplexity,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    return dimensionality_reduction

# Loader functions for all models we test
model_loaders = {
    "Topeax": lambda encoder, perplexity, vocabulary: Topeax(
        encoder=encoder,
        perplexity=perplexity,
        vectorizer=CountVectorizer(vocabulary=vocabulary),
        random_state=42,
    ),
    "BERTopic": lambda encoder, perplexity, vocabulary: BERTopic(
        encoder=encoder,
        dimensionality_reduction=create_umap(perplexity),
        vectorizer=CountVectorizer(vocabulary=vocabulary),
        random_state=42,
    ),
    "Top2Vec": lambda encoder, perplexity, vocabulary: Top2Vec(
        encoder=encoder,
        dimensionality_reduction=create_umap(perplexity),
        vectorizer=CountVectorizer(vocabulary=vocabulary),
        random_state=42,
    ),
}

PERPLEXITIES = [2, 5, 30, 50, 100]
ENCODER_NAME = "all-MiniLM-L6-v2"


def main():
    print("Loading external word embeddings")
    ex_wv = api.load("word2vec-google-news-300")
    ds = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
    )
    corpus = ds.data
    true_labels = ds.target
    encoder = SentenceTransformer(ENCODER_NAME)
    print("Training internal word embeddings using GloVe...")
    vectorizer = CountVectorizer().fit(corpus)
    vocabulary = vectorizer.vocabulary_
    print("Encoding vocab")
    vocab_embeddings = encoder.encode(
        vectorizer.get_feature_names_out(), show_progress_bar=True
    )
    tokenizer = vectorizer.build_analyzer()
    glove = GloVe(vector_size=50)
    tokenized_corpus = [tokenizer(text) for text in corpus]
    glove.train(tokenized_corpus)
    in_wv = glove.wv
    cache_dir = Path("embedding_cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir.joinpath("20news_all-MiniLM.npy")
    out_path = Path("results/robustness/perplexity.jsonl")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with out_path.open("w"):
        pass
    print("Trying to load embeddings from cahce.")
    if cache_file.is_file():
        embeddings = np.load(cache_file)
    else:
        print("Not found, encoding...")
        embeddings = encoder.encode(corpus, show_progress_bar=True)
        np.save(cache_file, embeddings)
    for model_name, loader in model_loaders.items():
        print(f"------------------Testing {model_name}--------------------")
        for perplexity in tqdm(PERPLEXITIES, desc="Going through perplexity values."):
            print(f"Running {model_name}(perplexity={perplexity}).")
            model = loader(encoder, perplexity, vocabulary)
            model.vocab_embeddings = vocab_embeddings
            model.fit(corpus, embeddings=embeddings)
            keywords = get_keywords(model)
            print("Evaluating model.")
            clust_scores = evaluate_clustering(true_labels, model.labels_)
            topic_scores = evaluate_topic_quality(keywords, ex_wv, in_wv)
            res = {
                "encoder": ENCODER_NAME,
                "model": model_name,
                "n_components": model.components_.shape[0],
                "true_n": len(set(true_labels)),
                "perplexity": perplexity,
                **clust_scores,
                **topic_scores,
            }
            print("Results: ", res)
            res["keywords"] = keywords
            with out_path.open("a") as out_file:
                out_file.write(json.dumps(res) + "\n")
    print("DONE")


if __name__ == "__main__":
    main()
