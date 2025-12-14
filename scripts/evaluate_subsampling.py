import json
from pathlib import Path
from typing import Optional

import gensim.downloader as api
import numpy as np
from evaluate_clustering import (evaluate_clustering, evaluate_topic_quality,
                                 get_keywords)
from glovpy import GloVe
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from turftopic import BERTopic, Top2Vec, Topeax

# Loader functions for each model
model_loaders = {
    "Topeax": lambda encoder, vocabulary: Topeax(
        encoder=encoder,
        vectorizer=CountVectorizer(vocabulary=vocabulary),
        random_state=42,
    ),
    "BERTopic": lambda encoder, vocabulary: BERTopic(
        encoder=encoder,
        vectorizer=CountVectorizer(vocabulary=vocabulary),
        random_state=42,
    ),
    "Top2Vec": lambda encoder, vocabulary: Top2Vec(
        encoder=encoder,
        vectorizer=CountVectorizer(vocabulary=vocabulary),
        random_state=42,
    ),
}

# None in this case means that we do not subsample
SAMPLE_SIZES = [250, 1000, 5000, 10_000, None]
ENCODER_NAME = "all-MiniLM-L6-v2"


def stratified_subsample(
    corpus, labels, embeddings, sample_size: Optional[int] = None, seed: int = 42
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Creates a subsample of the dataset stratified based on cluster labels"""
    if sample_size is None:
        # If None return the whole corpus
        return corpus, labels, embeddings
    # Ignoring "test" split
    corpus, _, labels, _, embeddings, _ = train_test_split(
        corpus,
        labels,
        embeddings,
        train_size=sample_size,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    return list(corpus), labels, embeddings


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
    print("Encoding vocab")
    vocab_embeddings = encoder.encode(
        vectorizer.get_feature_names_out(), show_progress_bar=True
    )
    word_to_embedding = dict(zip(vectorizer.get_feature_names_out(), vocab_embeddings))
    tokenizer = vectorizer.build_analyzer()
    glove = GloVe(vector_size=50)
    tokenized_corpus = [tokenizer(text) for text in corpus]
    glove.train(tokenized_corpus)
    in_wv = glove.wv
    cache_dir = Path("embedding_cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir.joinpath("20news_all-MiniLM.npy")
    out_path = Path("results/robustness/subsampling.jsonl")
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
        for sample_size in tqdm(SAMPLE_SIZES, desc="Going through subsampled corpora."):
            print(f"Running {model_name}(sample_size={sample_size}).")
            sub_corpus, sub_labels, sub_embeddings = stratified_subsample(
                corpus, true_labels, embeddings, sample_size=sample_size, seed=42
            )
            sub_vectorizer = CountVectorizer().fit(sub_corpus)
            sub_vocab = sub_vectorizer.vocabulary_
            sub_vocab_embeddings = [
                word_to_embedding[word]
                for word in sub_vectorizer.get_feature_names_out()
            ]
            sub_vocab_embeddings = np.stack(sub_vocab_embeddings)
            model = loader(encoder, sub_vocab)
            model.vocab_embeddings = sub_vocab_embeddings
            model.fit(sub_corpus, embeddings=sub_embeddings)
            keywords = get_keywords(model)
            print("Evaluating model.")
            clust_scores = evaluate_clustering(sub_labels, model.labels_)
            topic_scores = evaluate_topic_quality(keywords, ex_wv, in_wv)
            res = {
                "encoder": ENCODER_NAME,
                "model": model_name,
                "n_components": model.components_.shape[0],
                "true_n": len(set(sub_labels)),
                "sample_size": len(sub_corpus),
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
