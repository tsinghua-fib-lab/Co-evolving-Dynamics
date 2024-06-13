# Co-evolving-Dynamics


The repository contains the source codes for the manuscript "Emergence of modular structures in co-evolving networks". The source codes are implemented in python.
 
 # System Requirements

 ```
python == 3.8.11
numpy == 1.23.5
scipy == 1.10.1
scikit-learn ==  1.2.2
matplotlib == 3.7.1
seaborn == 0.12.2
jupyter notebook == 6.4.8
torch == 1.9.0+cu10
torch-geometric == 2.2.0
 ```
# Demo Guide

1. Install and enter the enviroment
2. Run simulation experiments in the one-dimensional scenario.

 ``` shell 
cd ./codes_1D/ 
sh run_1D.sh
 ```

3. Run simulation experiments in the high-dimensional scenario.

``` shell
cd ./codes_hD/
sh run_hD.sh
```

# Source Data

The source data used in the manuscript is available as follows:

1. Ecological dataset: [Web of Life](https://www.web-of-life.es/)
2. Information dataset: [Twitter](https://osf.io/e395q/) *
3. Urban dataset: [SafeGraph](https://www.safegraph.com/)

\* We acknowledge the authors for their open-sourced Twitter dataset.

Flamino, J., Galeazzi, A., Feldman, S., Macy, M.W., Cross, B., Zhou, Z., Serafino, M., Bovet, A., Makse, H.A. and Szymanski, B.K., 2023. Political polarization of news media and influencers on Twitter in the 2016 and 2020 US presidential elections. Nature Human Behaviour, 7(6), pp.904-916.

# File Description

* codes_1D: Folder for simulation experiments in one-dimensional scenario

    + Data: Data for random initialization
    + GPUSimulation.py: Script for simulation framework
    + run_experiments.py: Script for experimental setting
    + utils.py: Script for utilities
    + run_1D.sh: Script for running experiments


* codes_hD: Folder for simulation experiments in high-dimensional scenario

    + Data: Data for real-world initialization
    + GPUSimulation.py: Script for simulation framework
    + GPUSimulation_intervention.py: Script for simulation framework with external intervention
    + exp_2D.py: Script for experimental setting
    + exp_2D_intervention.py: Script for experimental setting
    + utils.py: Script for utilities
    + run_hD.sh: Script for running experiments