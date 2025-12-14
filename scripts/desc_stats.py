import numpy as np

from evaluate_clustering import load_corpora

# Generates a Typst Table of the descriptive statistics for each corpus
lines = []
for corpus_name, loader in load_corpora():
    documents, labels = loader()
    n_topics = len(set(labels))
    n_documents = len(documents)
    n_chars = [len(doc) for doc in documents]
    mean_chars = np.mean(n_chars)
    std_chars = np.std(n_chars)
    line = f"[{corpus_name}],[{mean_chars:.2f}±{std_chars:.2f}],[{n_documents}],[{n_topics}], "
    lines.append(line)

print("\n".join(lines))
