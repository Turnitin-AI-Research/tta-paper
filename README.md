This repository has the visualizaiton code for the paper "Analyzing Transformer Dynamics as Trajectories in Embedding Space".
All visualizations are in notebooks under the notebooks folder.

## Python environment setup
First ensure that you have python version 3.8. A higher version should work too, but in that case you'll need to recompile the requirements file tta.txt (explained later).
Then setup a python virtual environment called 'tta' in the directory .venv_tta as follows.
```
$ python3.8 -m venv --prompt tta .venv_tta
$ source .venv_tta/bin/activate
$ pip install pip-tools
$ pip-sync env_files/tta.txt
```
All notebooks and code runs under the above virtual environment.

`env_files/tta.in` is a dependency file where all the environment packages are listed. This gets compiled into a requirements file `tta.txt` which is used to populate the python virtual environment as above. If you choose to use a different version of python, or edit `tta.in`, then recreate `tta.txt` as below:

```
$ cd env_files
$ pip-compile tta.in > tta.txt
$ pip-sync tta.txt
```

### Optionally modify modeling_t5.py
If you want to run the notebook `notebooks/viz_attention_maps T5.ipynb`, then you'll need to modify some code in `.venv_tta/lib/python3.8/site-packages/transformers/models/t5/modeling_t5.py`. In order to do this, search for "tta-paper" in the file `notebooks/modeling_t5.py` - about 6 lines - and make the same changes in your transformers package's modeling_t5.py file.


## GPUs / Machine
We used a linux machine running Ubuntu 20.04 and having 8 Nvidia 2080-Ti GPUs. CUDA dependencies are automatically installed during the python environment setup.

## Vizualization Notebooks
The following notebook outputs were used in the paper:
1. Attention layer activations and position kernels:
    * `notebooks/viz_attention_maps T5.ipynb`
        * requires modifying T5 as explained above in order to extract pre-softmax attention-scores.
1. Similarity maps:
    * `notebooks/viz_similarity_maps T5.ipynb` for T5 and Flan-T5 models
    * `notebooks/viz_similarity_maps Falcon.ipynb` for the falcon-7b model

The remaining notebooks were not used for the paper and may not work reliably.

## figures folder
The notebooks write figures out to the notebooks/figures directory as PDF files.

## cache folder
The code will cache generated data in notebooks/cache folder
