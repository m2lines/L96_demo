---
title: 'Learning Machine Learning with Lorenz-96'
tags:
  - Python
  - Machine Learning
  - Neural Networks
  - Dynamical systems
authors:
  - name: Dhruv Balwada
    orcid: 0000-0001-6632-0187
    affiliation: 1
  - name: Ryan Abernathey
    orcid: 0000-0001-5999-4917
    affiliation: 1
  - name: Shantanu Acharya
    orcid: 0000-0002-9652-2991
    affiliation: 2
  - name: Alistair Adcroft
    orcid: 0000-0001-9413-1017
    affiliation: 3
  - name: Judith Brener
    orcid: 0000-0003-2168-0431
    affiliation: 12
  - name: V Balaji
    orcid: 0000-0001-7561-5438
    affiliation: 15
  - name: Mohamed Aziz Bhouri
    orcid: 0000-0003-1140-7415
    affiliation: 4
  - name: Joan Bruna
    orcid: 0000-0002-2847-1512
    affiliation: "2, 13"
  - name: Mitch Bushuk
    orcid: 0000-0002-0063-1465
    affiliation: 9
  - name: Will Chapman
    orcid: 0000-0002-0472-7069
    affiliation: 12
  - name: Alex Connolly
    orcid: 0000-0002-2310-0480
    affiliation: 4
  - name: Julie Deshayes
    orcid: 0000-0002-1462-686X
    affiliation: 10
  - name: Carlos Fernandez-Granda
    orcid: 0000-0001-7039-8606
    affiliation: "2, 13"
  - name: Pierre Gentine
    orcid: 0000-0002-0845-8345
    affiliation: "4, 14"
  - name: Anastasiia Gorbunova
    orcid: 0000-0002-3271-2024
    affiliation: 6
  - name: Will Gregory
    orcid: 0000-0001-8176-1642
    affiliation: 3
  - name: Arthur Guillaumin
    orcid: 0000-0003-1571-4228
    affiliation: 5
  - name: Shubham Gupta
    orcid: 0009-0002-6966-588X
    affiliation: 8
  - name: Marika Holland
    orcid: 0000-0001-5621-8939
    affiliation: 12
  - name: J Emmanuel Johnsson
    orcid: 0000-0002-6739-0053
    affiliation: 6
  - name: Julien Le Sommer
    orcid: 0000-0002-6882-2938
    affiliation: 6
  - name: Ziwei Li
    orcid:
    affiliation: 2
  - name: Nora Loose
    orcid: 0000-0002-3684-9634
    affiliation: 3
  - name: Feiyu Lu
    orcid: 0000-0001-6532-0740
    affiliation: 9
  - name: Paul O'Gorman
    orcid: 0000-0001-6532-0740
    affiliation: 11
  - name: Pavel Perezhogin
    orcid: 0000-0003-2098-3457
    affiliation: 3
  - name: Brandon Reichl
    orcid: 0000-0001-9047-0767
    affiliation: 9
  - name: Andrew Ross
    orcid: 0000-0002-2368-6979
    affiliation: 2
  - name: Aakash Sane
    orcid: 0000-0002-9642-008X
    affiliation: 3
  - name: Sara Shamekh
    orcid: 0000-0001-7441-4116
    affiliation: 4
  - name: Tarun Verma
    orcid: 0000-0001-7730-1483
    affiliation: 3
  - name: Janni Yuval
    orcid: 0000-0001-7519-0118
    affiliation: 11
  - name: Lorenzo Zampieri
    orcid: 0000-0003-1703-4162
    affiliation: 7
  - name: Cheng Zhang
    orcid: 0000-0003-4278-9786
    affiliation: 3
  - name: Laure Zanna
    orcid: 0000-0002-8472-4828
    affiliation: 2
affiliations:
  - name: Lamont Doherty Earth Observatory, Columbia University
    index: 1
  - name: Courant Institute of Mathematical Sciences, New York University
    index: 2
  - name: Program in Atmospheric and Oceanic Sciences, Princeton University
    index: 3
  - name: Earth and Environmental Engineering, Columbia University
    index: 4
  - name: Queen Mary University of London
    index: 5
  - name: Univ. Grenoble Alpes, CNRS, IRD, Grenoble INP, INRAE, IGE, 38000 Grenoble, France
    index: 6
  - name: Ocean Modeling and Data Assimilation Division, Fondazione Centro Euro-Mediterraneo sui Cambiamenti Climatici - CMCC
    index: 7
  - name: Tandon School of Engineering, New York University
    index: 8
  - name: NOAA Geophysical Fluid Dynamics Laboratory
    index: 9
  - name: Sorbonne Universités, LOCEAN Laboratory, Paris, France
    index: 10
  - name: Department of Earth, Atmospheric, and Planetary Sciences, Massachusetts Institute of Technology
    index: 11
  - name: National Center for Atmospheric Research
    index: 12
  - name: Center for Data Science, New York University
    index: 13
  - name: Columbia Climate School, Columbia University
    index: 14
  - name: Schmidt Futures
    index: 15
date: 10 October 2023

bibliography: paper.bib
---
# Summary
Machine learning (ML) is a rapidly growing field that is starting to touch all aspects of our lives, and science is not immune to this. In fact, recent work in the field of scientific ML, ie combining ML and with conventional scientific problems, is leading to new breakthroughs in notoriously hard problems, which might have seemed too distant till a few years ago. One such age-old problem is that of turbulence closures in fluid flows (described later). This closure or parameterization problem is particularly relevant for environmental fluids, which span a large range of scales from the size of the planet down to millimeters, and remains a big challenge in the way of improving forecasts of weather and projections of climate.

The climate system is composed of many interacting components (e.g., ocean, atmosphere, ice) and is described by complex nonlinear equations. To simulate, understand and predict climate, these equations are solved numerically under a number of simplifications, therefore leading to errors. The errors result from numerics used to solve the equations and the lack of appropriate representations of processes occurring below the resolution of the climate model grid (i.e., subgrid processes).

This book aims to conceptualize the problems associated with climate models within a simple and computationally accessible framework, and show how some basic ML methods can be used to approach these problems. We will introduce the readers to climate modeling using a simple tool, the @Lorenz1995 (L96) two-timescale model. We discuss the numerical aspects of the L96 model, the approximate representation of subgrid processes (known as parameterizations or closures), and simple data assimilation problems (a data-model fusion method). We then use the L96 results to demonstrate how to learn subgrid parameterizations from data with machine learning, and then test the parameterizations offline (apriori) and online (aposteriori), with a focus on the interpretability of the results. This book is written primarily for climate scientists and physicists, who are looking for a gentle introduction to how they can incorporate machine learning into their work. However, it may also help machine learning scientists learn about the parameterization problem in a framework that is relatively simple to understand and use.

The material in this Jupyter book is presented over five sections. The first section, Lorenz 96 and General Circulations Models, describes the Lorenz-96 model and how it can work as a simple analog to much more complex general circulation models used for simulating ocean and atmosphere dynamics. This section also introduces the essence of the parameterization or closure problem. In the second section, Neural Networks with Lorenz-96, we introduce the basics of machine learning, how fully connected neural networks can be used to approach the parameterization task, and how these neural networks can be optimized and interpreted. No model, even the well parameterized ones, is perfect, and the way we keep computer models close to reality is by guiding them with the help of observational data. This task is referred to as data assimilation, and is introduced in the third section, Data Assimilation with Lorenz-96. Here, we use the L96 model to quickly introduce the concepts from data assimilation, and show how machine learning can be used to learn data assimilation increments to help reduce model biases. While neural networks can be great functional approximators, they are usually quite opaque, and it is hard to figure out exactly what they have learnt. Equation discovery is a class of machine learning techniques that tries to estimate the function in terms of an equation rather than as a set of weights for a neural network. This approach produces a result that is far more interpretable, and can potentially even help discover novel physics. These techniques are presented in the fourth section, Equation Discovery with Lorenz-96. Finally, we describe a few more machine learning in section five, Other ML approaches for Lorenz-96, with the acknowledgment that there are many more techniques in the fast-growing ML and scientific ML literature and we have no intention of providing a comprehensive summary of the field.

The book was created by and as part of M2LInES, an international collaboration supported by Schmidt Futures, to improve climate models with scientific machine learning. The original goal for these notebooks in this Jupyter book was for our team to work together and learn from each other; in particular, to get up to speed on the key scientific aspects of our collaboration (parameterizations, machine learning, data assimilation, uncertainty quantification) and to develop new ideas. This was done as a series of tutorials, each of which was led by a few team members and occurred with a frequency of roughly once every 2 weeks for about 6-7 months. This Jupyter book is a collection of the notebooks used during these tutorials, which have only slightly been edited for continuity and clarity. Ultimately, we are happy to share these resources with the scientific community to introduce our research ideas and foster the use of machine learning techniques for tackling climate science problems.

# Statement of Need
Parameterization of sub-grid processes is a major challenge in climate science. The exact details of this problem are often very context dependent (@christensen2022parametrization), ... 

As described above, these notebooks were originally created to introduce non-domain experts to ideas from the parameterization aspects of climate modeling and how machine learning could be used to potentially address these. Now they have been adapted to act as a pedagogical tool for self-learning, be used as a reference manual, or for teaching some modules in an introductory class on machine learning. 
The book is organized in sections that are relatively independent; with the exception that the first section provides a general overview to the parameterization problem in climate models. 
Each notebook covers material that can be discussed in roughly an hour-long lecture, and sections can be mixed and matched or ordered as needed depending on the overall learning objectives.


# Acknowledgements
This work is supported by the generosity of Eric and Wendy Schmidt by recommendation of Schmidt Futures, as part of its Virtual Earth System Research Institute (VESRI). MAB acknowledges support from National Science Foundation's AGS-PRF Fellowship Award (AGS2218197).

# References
