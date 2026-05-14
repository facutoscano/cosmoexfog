# CosmoExFog 

**Cosmological Parameters derived from CMB Extragalactic Foregrounds**

**CosmoExFog** is a comprehensive Python repository designed for the estimation of cosmological parameters from Cosmic Microwave Background (CMB) data. Its primary focus is to quantify how the full CMB spectrum—and consequently the derived parameters—are influenced by masking [local extragalactic foregrounds](https://ui.adsabs.harvard.edu/public-libraries/rsJl_F4IRqOJ_2eP8I07jA).

### Key Features:
* **Flexible Masking & Power Spectra:** Extract pseudo-$C_\ell$ and compute bandpowers from full-sky maps applying custom galaxy-derived masks.
* **Scale-Dependent Discrepancy Analysis:** Perform systematic multipole cuts ($l_{min}$ and $l_{max}$) to estimate parameter discrepancies across different angular scales.
* **Dual Inference Engines:** * Fast profile likelihood minimization using **[`iminuit`](https://iminuit.readthedocs.io/)**.
  * Rigorous Markov Chain Monte Carlo (MCMC) posterior sampling using **[`Cobaya`](https://cobaya.readthedocs.io/)**.
* **Simulation-Based Sanity Checks:** Pipeline validation and covariance matrix estimation utilizing Planck mock realizations.

This framework allows users to thoroughly investigate parameter shifts and large-scale anomalies, providing the tools to analyze whether current $H_0$ or $\Omega_m$ tensions could be partially biased by unmodeled local foregrounds.
