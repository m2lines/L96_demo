# Lorenz 1996 two time-scale model

[![build-and-deploy-book](https://github.com/m2lines/L96_demo/actions/workflows/deploy.yml/badge.svg)](https://github.com/m2lines/L96_demo/actions/workflows/deploy.yml)

## Structure and Organization of the Repo

This project uses [Jupyter Book](https://jupyterbook.org/) to organize a collection of
Jupyter Notebooks into a website.

- The notebooks all live in the [notebooks](https://github.com/m2lines/L96_demo/tree/main/notebooks) directory.
  Note that the notebooks are stored in "stripped" form, without any outputs of execution saved.
  (They are executed as part of the build process.)
- The table of contents is located in [\_toc.yml](https://github.com/m2lines/L96_demo/blob/main/_toc.yml).
- The book configuration is in [\_config.yml](https://github.com/m2lines/L96_demo/blob/main/_config.yml).
- The references are in [\_references.bib](https://github.com/m2lines/L96_demo/blob/main/references.bib).

## The Environment

The environment in which to run the notebooks and build the books is defined in
[environment.yaml](https://github.com/m2lines/L96_demo/blob/main/environment.yaml).
To recreate and activate the environment locally, run

```
conda env create -f environment.yaml
conda activate L96M2lines
```

To speed up the continuous integration, we also generated a
[conda lock](https://conda.github.io/conda-lock/) file for linux as follows.

```
conda-lock lock --mamba -f environment.yaml -p linux-64
```

This file lives in [conda-linux-64.lock](https://github.com/m2lines/L96_demo/blob/main/conda-linux-64.lock).
It should be regenerated periorically.

## Building the Book

To build the book locally, you should first create and activate your environment,
as described above. Then run

```
jupyter book build .
```

When you run this command, the notebooks will be executed.
The built html will be placed in '\_build/html`.
To preview the book, run

```
cd _build/html
python -m http.server
```

The build process can take a long time, so we have configured the setup to use
[jupyter-cache](https://jupyter-cache.readthedocs.io/en/latest/).
If you re-run the `build` command, it will only re-execute notebooks
that have been changed. The cache files live in `_build/.jupyter_cache`

To check the status of the cache, run

```
jcache cache list -p _build/.jupyter_cache
```

To clear the cache, run

```
jcache cache clear -p _build/.jupyter_cache
```

## Contributing

### Pre-commit

We use [pre-commit](https://pre-commit.com/) to keep the notebooks clean.
In order to use pre-commit, run the following command in the repo top-level directory:
The pre commit

```
pre-commit install
```

At this point, pre-commit will automatically be run every time you make a commit.

### Pull Requests and Feature Branches

In order to contribute a PR, you should start from a new feature branch.

```
git checkout -b my_new_feature
```

(Replace `my_new_feature` with a descriptive name of the feature you're working on.)

Make your changes and then make a new commit:

```
git add changed_file_1.ipynb changed_file_2.ipynb
git commit -m "message about my new feature"
```

You can also automatically commit changes to existing files as:

```
git commit -am "message about my new feature"
```

Then push your changes to your remote on GitHub (usually call `origin`

```
git push origin my_new_feature
```

Then navigate to https://github.com/m2lines/L96_demo to open your pull request.

### Synchronizing from upstream

To synchronize your local branch with upstream changes, first make sure you have the upstream remote configured.
To check your remotes, run

```
% git remote -v
origin	git@github.com:rabernat/L96_demo.git (fetch)
origin	git@github.com:rabernat/L96_demo.git (push)
upstream	git@github.com:m2lines/L96_demo.git (fetch)
upstream	git@github.com:m2lines/L96_demo.git (push)
```

If you don't have `upstream`, you need to add it as follows

```
git remote add upstream git@github.com:m2lines/L96_demo.git
```

Then, make sure you are on the main branch locally:

```
git checkout main
```

And then run

```
git fetch upstream
git merge upstream/main
```

Ideally you will not have any merge conflicts.
You are now ready to make a new feature branch.

## References

Arnold, H. M., I. M. Moroz, and T. N. Palmer. “Stochastic Parametrizations and Model Uncertainty in the Lorenz ’96 System.” Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences 371, no. 1991 (May 28, 2013): 20110479. https://doi.org/10.1098/rsta.2011.0479.

Brajard, Julien, Alberto Carrassi, Marc Bocquet, and Laurent Bertino. “Combining Data Assimilation and Machine Learning to Infer Unresolved Scale Parametrization.” Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences 379, no. 2194 (April 5, 2021): 20200086. https://doi.org/10.1098/rsta.2020.0086.

Schneider, Tapio, Shiwei Lan, Andrew Stuart, and João Teixeira. “Earth System Modeling 2.0: A Blueprint for Models That Learn From Observations and Targeted High-Resolution Simulations.” Geophysical Research Letters 44, no. 24 (December 28, 2017): 12,396-12,417. https://doi.org/10.1002/2017GL076101.

Wilks, Daniel S. “Effects of Stochastic Parametrizations in the Lorenz ’96 System.” Quarterly Journal of the Royal Meteorological Society 131, no. 606 (2005): 389–407. https://doi.org/10.1256/qj.04.03.
