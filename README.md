# Lorenz 1996 two time-scale model

## Building the Jupyter Book locally

```
pip install requirements.txt
jupyter book build .
cd _build/html
python -m http.server
```

## Contents in 01Intro:
- L96-description.ipynb : Equation and demonstration of the single time-scale model
- L96-two-scale-description.ipynb : Equations and demonstration of the two time-scale model
- L96_model.py : Functions providing tendancies and integrators for the L96 models

## Required packages

- jupyter (for notebooks)
- numpy (used in computations)
- matplotlib (for plots)
- numba (for significant speed up)

### Conda

If you are starting from scratch, install conda and then:
```bash
conda create -n py3 jupyter numpy matplotlib
conda activate py3
jupyter notebook
```

## References

Arnold, H. M., I. M. Moroz, and T. N. Palmer. “Stochastic Parametrizations and Model Uncertainty in the Lorenz ’96 System.” Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences 371, no. 1991 (May 28, 2013): 20110479. https://doi.org/10.1098/rsta.2011.0479.

Brajard, Julien, Alberto Carrassi, Marc Bocquet, and Laurent Bertino. “Combining Data Assimilation and Machine Learning to Infer Unresolved Scale Parametrization.” Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences 379, no. 2194 (April 5, 2021): 20200086. https://doi.org/10.1098/rsta.2020.0086.

Schneider, Tapio, Shiwei Lan, Andrew Stuart, and João Teixeira. “Earth System Modeling 2.0: A Blueprint for Models That Learn From Observations and Targeted High-Resolution Simulations.” Geophysical Research Letters 44, no. 24 (December 28, 2017): 12,396-12,417. https://doi.org/10.1002/2017GL076101.

Wilks, Daniel S. “Effects of Stochastic Parametrizations in the Lorenz ’96 System.” Quarterly Journal of the Royal Meteorological Society 131, no. 606 (2005): 389–407. https://doi.org/10.1256/qj.04.03.
