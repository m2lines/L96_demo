# Lorenz 1996 two time-scale model

[![build-and-deploy-book](https://github.com/m2lines/L96_demo/actions/workflows/deploy.yml/badge.svg)](https://github.com/m2lines/L96_demo/actions/workflows/deploy.yml)

## Building the Jupyter Book locally

```
conda env create -f environment.yaml
conda activate L96M2lines
mkdir -p _notebook_cache
jupyter book build .
cd _build/html
python -m http.server
```

## Contributing

### Pre-commit

We use [pre-commit](https://pre-commit.com/) to keep the notebooks clean.
In order to use pre-commit, run the following command in the repo top-level directory:

```
pre-commit install
```

At this point, pre-commit will automatically be run every time you make a commit.


## References

Arnold, H. M., I. M. Moroz, and T. N. Palmer. “Stochastic Parametrizations and Model Uncertainty in the Lorenz ’96 System.” Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences 371, no. 1991 (May 28, 2013): 20110479. https://doi.org/10.1098/rsta.2011.0479.

Brajard, Julien, Alberto Carrassi, Marc Bocquet, and Laurent Bertino. “Combining Data Assimilation and Machine Learning to Infer Unresolved Scale Parametrization.” Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences 379, no. 2194 (April 5, 2021): 20200086. https://doi.org/10.1098/rsta.2020.0086.

Schneider, Tapio, Shiwei Lan, Andrew Stuart, and João Teixeira. “Earth System Modeling 2.0: A Blueprint for Models That Learn From Observations and Targeted High-Resolution Simulations.” Geophysical Research Letters 44, no. 24 (December 28, 2017): 12,396-12,417. https://doi.org/10.1002/2017GL076101.

Wilks, Daniel S. “Effects of Stochastic Parametrizations in the Lorenz ’96 System.” Quarterly Journal of the Royal Meteorological Society 131, no. 606 (2005): 389–407. https://doi.org/10.1256/qj.04.03.
