## Large Scale Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple Methods

[*Large Scale Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple Methods*](https://arxiv.org/abs/2110.14446)


Derek Lim*, Felix Hohne*, Xiuyu Li*, Sijia Linda Huang, Vaishnavi Gupta, Omkar Bhalerao, Ser-Nam Lim  

Published at NeurIPS 2021

Here are codes to load our proposed datasets, compute our measure of homophily, and train various graph machine learning models in our experimental setup. We include an implementation of the new graph neural network LINKX that we develop.

## Organization
`main.py` contains the main full batch experimental scripts.

`main_scalable.py` contains the minibatching experimental scripts.

`parse.py` contains flags for running models with specific settings and hyperparameters. 

`dataset.py` loads our datasets.

`models.py` contains implementations for graph machine learning models, though C&S (`correct_smooth.py`, `cs_tune_hparams.py`) are in separate files. Running several of the GNN models on larger datasets may require at least 24GB of VRAM. **Our LINKX model is implemented in this file.**

`homophily.py` contains functions for computing homophily measures, including the one that we introduce in `our_measure`.

`experiments/` contains the bash files to reproduce full batch experiments. 

`scalable_experiments/` contains the bash files to reproduce minibatching experiments. 

`wiki_scraping/` contains the Python scripts to reproduce the "wiki" dataset by querying the Wikipedia API and cleaning up the data.

## Datasets

<img width="2419" alt="Screenshot 2021-06-03 at 6 04 01 PM" src="https://user-images.githubusercontent.com/58995473/120717799-27fd9200-c496-11eb-940f-b16d4d528e0f.png">



As discussed in the paper, our proposed datasets are "genius", "twitch-gamer", "fb100", "pokec", "wiki", "arxiv-year", and "snap-patents", which can be loaded by `load_nc_dataset` in `dataset.py` by passing in their respective string name. Many of these datasets are included in the `data/` directory, but wiki, twitch-gamer, snap-patents, and pokec are automatically downloaded from a Google drive link when loaded from `dataset.py`. The arxiv-year dataset is downloaded using OGB downloaders. `load_nc_dataset` returns an NCDataset, the documentation for which is also provided in `dataset.py`. It is functionally equivalent to OGB's Library-Agnostic Loader for Node Property Prediction, except for the fact that it returns torch tensors. See the [OGB website](https://ogb.stanford.edu/docs/nodeprop/) for more specific documentation. Just like the OGB function, dataset.get_idx_split() returns fixed dataset split for training, validation, and testing. 

When there are multiple graphs (as in the case of fb100), different ones can be loaded by passing in the `sub_dataname` argument to `load_nc_dataset` in `dataset.py`. In particular, fb100 consists of 100 graphs. We only include ["Amherst41", "Cornell5", "Johns Hopkins55", "Penn94", "Reed98"] in this repo, although others may be downloaded from [the internet archive](https://archive.org/details/oxford-2005-facebook-matrix). In the paper we test on Penn94.

## References

The datasets come from a variety of sources, as listed here:
* **Penn94**. Traud et al 2012. _Social Structure of Facebook_ Networks
* **pokec**. Leskovec et al. _Stanford Network Analysis Project_
* **arXiv-year**. Hu et al 2020. _Open Graph Benchmark_
* **snap-patents**. Leskovec et al. _Stanford Network Analysis Project_
* **genius**. Lim and Benson 2020. _Expertise and Dynamics within Crowdsourced Musical Knowledge Curation: A Case Study of the Genius Platform_
* **twitch-gamers**. Rozemberczki and Sarkar 2021. _Twitch Gamers: a Dataset for Evaluating Proximity Preserving and Structural Role-based Node Embeddings_
* **wiki**. Collected by the authors of this work in 2021. The full details are available in Appendix D.3.

## Installation instructions

1. Create and activate a new conda environment using python=3.8 (i.e. `conda create --name non-hom python=3.8`) 
2. Activate your conda environment
3. Check CUDA version using `nvidia-smi` 
4. run `bash install.sh cu110`, replacing cu110 with your CUDA version (CUDA 11 -> cu110, CUDA 10.2 -> cu102, CUDA 10.1 -> cu101). We tested on Ubuntu 18.04, CUDA 11.0.


## Running experiments

1. Make sure a results folder exists in the root directory. 
2. Our experiments are in the `experiments/` and `scalable_experiments/` directory. There are bash scripts for running methods on single and multiple datasets. Please note that the experiments must be run from the root directory, e.g. (`bash experiments/mixhop_exp.sh snap-patents`). For instance, to run the MixHop experiments on arxiv-year, use:
```
bash experiments/mixhop_exp.sh arxiv-year
```

To run LINKX on pokec, use: 

```
bash experiments/linkx_exp.sh pokec
```

To run LINK on Penn94, use: 

```
bash experiments/link_exp.sh fb100 Penn94
```

To run GCN-cluster on twitch-gamers, use: 

```
bash scalable_experiments/gcn_cluster.sh twitch-gamer
```

To run LINKX minibatched on wiki, use 

```
bash scalable_experiments/linkx_exp.sh wiki
```

To run LINKX on Geom-GCN with full hyperparameter grid on chameleon, use 
```
bash experiments/linkx_tuning.sh chameleon
```
