This repository has the visualizaiton code for the paper "Analyzing Transformer Dynamics as Trajectories in Embedding Space".
All visualizations are in notebooks under the notebooks folder.

## Python environment setup
First ensure that you have python version >=3.8.13 .
Then setup a python virtual environment called 'tta' in the directory .venv_tta as follows.
```
$ python3 -m venv --prompt tta .venv_tta
$ source .venv_tta/bin/activate
$ pip install pip-tools
$ pip-sync env_files/tta.txt
```
All notebooks and code runs under the above virtual environment.

`env_files` has files tta.in where all the environment packages are listed. This gets compiled into a requirements file: tta.txt which is
used to populate the python virtual environment as above. Should you need to recreate the environment file tta.txt follow these instructions:

```
$ cd env_files
$ pip-compile tta.in > tta.txt
$ pip-sync tta.txt
```

## GPUs Machine
We used a linux machine running Ubuntu 20.04 and having 8 Nvidia 2080-Ti GPUs. CUDA dependencies are automatically installed during the python environment setup (see above).

## Vizualization Notebooks
The following notebook outputs were used in the paper:
1. Attention layer activations and position kernels:
    * "notebooks/viz_attention_maps T5.ipynb"
1. Similarity maps:
    * "notebooks/viz_similarity_maps T5.ipynb" for T5 and Flan-T5 models
    * "notebooks/viz_similarity_maps Falcon.ipynb" for the falcon-7b model

The remaining notebooks were not used for the paper and may not work reliably.


## figures folder
The notebooks write figures out to the notebooks/figures directory as PDF files.

## cache folder
The code will cache generated data in notebooks/cache folder
