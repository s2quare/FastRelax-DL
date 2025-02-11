# FastRelax-DL
**Flexible and Cost-Effective Deep Learning for Accelerated Multi-Parametric Relaxometry using Phase-Cycled bSSFP**

![Figure1](figures/fig1.png)
The workflow proposed in this work (purple cubes represent the extended input and output in case of complex-based DNNs). (**a**) **Data Simulation:** The input parameters **$\text{p} = \left\{  T_{1}, T_{2}, B_1^+, \Delta B_0 \right\}$** entering the analytical bSSFP signal model (see \autoref{eq:1}) were sampled from three different distributions (in vivo, uniform, and uniform extended) for $T_1$ and $T_2$, and from a single uniform distribution for $B_1^+$ and $\Delta B_0$. The sequence parameters from the in vivo acquisition protocol (TR, TE, $\alpha_{nom}$, $N_{pc}$) were used to draw 400,000 signal samples $S_{\text{bSSFP}}$ from each $T_1$ and $T_2$ distribution. (**b**) **Multi-Parametric-Fitting Frameworks:** The input to each of the three frameworks, which means the physics-informed neural network (PINN or $PINN_{complex}$, 1), the supervised neural network (SVNN or $SVNN_{complex}$, 2), and the iterative golden section search (GSS) fitting (MIRACLE, 3), consisted of the amplitudes (magnitude-based) or real and imaginary parts (complex-based, without imaginary part of $F_0$) of the three lowest-order SSFP configurations computed from a Fourier transform (FT) of the phase-cycled bSSFP signal with the option to add noise and in addition of $B_1^+$. 1) and 2) use the same multilayer perceptron architecture (magnitude-based: 64 neurons per hidden layer, complex-based: 256 neurons per hidden layer) to estimate the inverse signal model and predict the parameters **$\hat{\text{p}} \in \left\{\hat{T}_{1}, \hat{T}_{2}, \hat{\Delta{B}}_0\right\}$**. 1) uses the predicted $\hat{T}_1$, $\hat{T}_2$, and $\hat{\Delta B_0}$ (with the addition of the $B_1^+$ input) to generate an estimated signal $\hat{S}$ and compare it to the input signal $S_{\text{inp}}$ in the $L_{\text{PINN}}$ loss, while 2) compares the predicted $\hat{T}_1$, $\hat{T}_2$, and $\hat{\Delta B_0}$ directly to the respective ground truth target parameters **$\text{p} \in \left\{T_{1}, T_{2}, \Delta{B_0}\right\}$** in the $L_{\text{SVNN}}$ loss. The off-resonance $\Delta{B_0}$ was only utilized for the complex-based DNNs (purple cubes).

# Summary 

This repository contains all necessary code to create the deep neural network (DNN) in silico training data, train the DNNs using PyTorch, and generate the code-based versions of the figures of the paper. The repository is structured as follows:

- Simulate the in silico training data using `scripts/in_silico_create_training_data.py` 
    - The respective configs file for the in silico distribution can be found in `cfgs/0.simulation/cfg_*.yml`
- Train the DNNs based on in silico data using `scripts/train_in_silico.py`
    - The respective config file for the training can be found in `cfgs/1.nn_train/train_in_silico.yml`
- Generate the figures of the paper using `scripts/figure*.py`
    - Generated Figures are based on code and might not exactly match the figures in the paper
- The trained models from the paper can be found in `models/`
    - `magnitude-based/trained-snr`: All DNN modesl trained with different SNRs
    - `magnitude-based`: The magnitude-based DNNs trained in the second iteration
    - `complex-based`: The complex-based DNNs used for Figure 5, 6, and 7
    - `fastrelax`: Starting a new training will save the new model weights here
    - `fig8`: Contains all models weights for each epoch to create Figure 8
- The scripts to process raw in vivo data are not shared but already prepared in vivo test data for DNNs can be requested.

# Paper
[Flexible and Cost-Effective Deep Learning for Accelerated Multi-Parametric Relaxometry using Phase-Cycled bSSFP](https://link.springer.com/article/10.1038/s41598-025-88579-z?utm_source=rct_congratemailt&utm_medium=email&utm_campaign=oa_20250209&utm_content=10.1038%2Fs41598-025-88579-z).

```
@article{birk_flexible_2025,
	title = {Flexible and cost-effective deep learning for accelerated multi-parametric relaxometry using phase-cycled {bSSFP}},
	volume = {15},
	issn = {2045-2322},
	url = {https://doi.org/10.1038/s41598-025-88579-z},
	doi = {10.1038/s41598-025-88579-z},
	number = {1},
	journal = {Scientific Reports},
	author = {Birk, Florian and Mahler, Lucas and Steiglechner, Julius and Wang, Qi and Scheffler, Klaus and Heule, Rahel},
	month = feb,
	year = {2025},
	pages = {4825},
}
```

# Installation
The conda environment (FastRelax-DL) used in this work can be installed using:
```bash
conda env create -f env.yml
```
miniforge is recommended for installation (using only the conda-forge channel)


# Data 
Please note that in silico and in vivo data to reproduce all Figures can be requested from the authors upon reasonable request. Please contact florian.birk@tuebingen.mpg.de

In case you downloaded the data copy the `in_silico` and `in_vivo` folders to the `path/to/project/FastRelax-DL/data` folder. The data folder should look like this:
- `in_silico/`
    - `mc`: Results from in silico MC simulations
    - `pred`: Results from the DNN predictions of Figure 5
    - `test`: Test data for the DNNs
    - `train`: Training data for the DNNs
- `in_vivo/`
    - `density_map`: Density maps to create the density distributions plots of Figure 1
    - `metrics`: Metrics calculated for each epoch of Figure 8
    - `pred`: Results from the DNN predictions of Figure 6 and 7
    - `test`: Preprocessed and prepared PyTorch input datasets for the DNNs, a converted version in nifti format, and the Masks used for the Figures. 


# Licensing
This repository is licensed under the MIT License. For more information, please see the LICENSE file.

`Note: The paper associated with this repository is licensed under a separate license, Creative Commons Attribution 4.0 (CC BY 4.0). If you are using or referencing the paper, please ensure you comply with the terms of the CC BY 4.0 license.`