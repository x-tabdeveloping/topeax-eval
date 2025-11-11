# topeax-eval

Evaluating the Topeax topic model for my NLP exam project for the Cognitive Science master programme at Aarhus University.
This repository contains evaluation scripts and results for the paper.

<img src="figures/peax.png">

## Installation

In order to reproduce evaluations, figures and tables, one first has to fetch the repository and install the required packages.

```bash
git clone https://github.com/x-tabdeveloping/topeax-eval
cd topeax-eval
pip install -r requirements.txt
```

## Reproducing results

To run evaluations, three separate scripts will have to be used.
`evaluate_clustering.py` runs the main evaluations, while `evaluate_perplexity_robustness.py` and `evaluate_subsampling.py` run the smaller, robustness experiments in sections 3.5 and 3.4.

```bash
python scripts/evaluate_clustering.py "all-MiniLM-L6-v2"
python scripts/evaluate_clustering.py "all-mpnet-base-v2"
python scripts/evaluate_clustering.py "google/embeddinggemma-300m"

python scripts/evaluate_perplexity_robustness.py
python scripts/evaluate_subsampling.py
```

## Reproducing tables and figures

Here's a list of all scripts used to extract data for tables in the paper, as well as scripts to generate the plots:

| Script | Description |
| ------ | ----------- |
| `python scripts/desc_stats.py` | Prints descriptive statistics for datasets used in the evaluations. |
| `python scripts/plot_performance.py` | Creates the main performance plot (Interpretability against FMI, and percentage error in No. topics) |
| `python scripts/qualitative_20ng.py` | Creates plots in section 4.3 and prints information for BERTopic and Top2Vec. |
| `python scripts/regression_modeling.py` | Runs the mixed effects model from section 4 and prints results in CSV format. |
| `python scripts/robustness_plotting.py` | Generates plots in sections 4.1 and 4.2 in the paper on model performance against sample size and perplexity. |
