# Mechanistic pathway modeling reveals how IL-10 generates pleiotropic immune responses

<p align="center">
  <img src="assets/comp_framework.jpg" alt="Graphical abstract of the IL-10 modeling framework" width="950">
</p>

This repository contains the computational code used in the manuscript:

**“Mechanistic pathway modeling reveals how IL-10 generates pleiotropic immune responses.”**

The study combines RNA-to-protein linear models, mechanistic ordinary differential equation (ODE) models of IL-10 signaling, Bayesian model selection and parameter inference, dose-response simulations, and transcriptomic analyses to investigate how IL-10 receptor-binding kinetics and cell-type-specific receptor/STAT abundance shape STAT1/STAT3 signaling and downstream gene regulation.

**Preprint:** https://doi.org/10.64898/2026.06.02.729479  
**Code:** https://github.com/Quim98/IL10_model_publication  
**Processed data and simulation outputs:** https://doi.org/10.5281/zenodo.22257307
**Raw proteomics:** PRIDE accession **PXD076154**

---

## Reproducibility overview

GitHub repository contains the analysis, inference, simulation, and plotting code. Large processed datasets, model initial conditions, and simulation/inference outputs are hosted separately on Zenodo.

For most manuscript figures, it is **not necessary to rerun the computationally expensive ABC-SMC inference**: the required processed data and precomputed outputs are included in the Zenodo archive. To reproduce the figures, clone this repository, extract the Zenodo archive into the repository root while preserving its directory structure, create the supplied environment, and run the corresponding plotting notebook.

To regenerate the model outputs from the underlying processed data, the inference and simulation scripts described below can instead be rerun before executing the plotting notebooks.

---

## Repository structure

The main code files are organized as follows:

```text
IL10_model_publication/
├── environment.yaml
│
├── src/
│   └── shared model definitions, simulation functions, and analysis utilities
│
├── model_perturbations/
│   └── in-vitro dose-response simulation scripts on modified mechanistic models
│
├── pyabc_inference_WT_ODE.py
├── pyabc_inference_mut_ODE.py
├── pyabc_inference_mut_ODE_IL10M.py
├── pyabc_model_selection.py
│   └── ABC-SMC parameter inference/model selection scripts
│
├── sim_invitro_fit_eIC.py
├── sim_invitro_fit_eIC_ODE.py
├── sim_invitro_IL10M_eIC.py
├── sim_invitro_scIL10_eIC.py
│   └── in-vitro dose-response simulation scripts
│
├── sim_cells_cytk_dict.py
│   └── simulation scripts for immune cells in Immune dictionary
│
├── slopiness_analysis.py
│   └── parameter-sensitivity/sloppiness analysis
│
├── Plots_linear_models.ipynb
├── Plots_slopiness_analysis.ipynb
├── Plots_ABC_SMC.ipynb
├── Plots_dose_response.ipynb
├── Mech_models_performance.ipynb
├── Null_model_IL10.ipynb
├── Plots_genome_regulation.ipynb
    └── analysis and manuscript-figure notebooks

```

The directories below are supplied in the **Zenodo archive** and should be placed in the repository root:

```text
SS_search/
data/
results/
```

### Zenodo data structure

| Directory | Contents |
|---|---|
| `SS_search/` | Pickle/intermediate files used to obtain simplified steady states with the linear-framework implementation of the signaling models. |
| `data/ABC_SMC/` | Input data, initial parameter values, bounds, and other files used for ABC-SMC model selection and parameter inference. |
| `data/binding/` | IL-10/receptor and signaling binding-rate parameters, including trained and literature/untrained values. |
| `data/expression/` | Bulk RNA-seq/proteomics data used to train and validate the RNA-to-protein models and to define receptor/STAT initial conditions for signaling simulations. |
| `data/lin_models/` | Parameters and outputs of the gene-specific RNA-to-protein linear models. |
| `data/signaling/` | Experimental dose-response data used to train/test the IL-10 signaling models and RNA-seq datasets used in the downstream gene-regulation analyses. |
| `data/uniprot/` | UniProt identifiers and associated mapping information. |
| `results/ABC_SMC/` | ABC-SMC model-selection and parameter-inference outputs. |
| `results/immune_dict/` | Receptor Memory model simulations for Immune Dictionary cell types, including STAT1/STAT3 predictions used in the gene-regulation analyses. |
| `results/raw_param/` | Simulations obtained using the non-fitted/literature parameter sets. |
| `results/` | Additional fitted-model simulation and sloppiness-analysis outputs used by the plotting notebooks. Preserve the directory structure provided in the Zenodo archive. |

---

## Software environment

The complete computational environment is specified in `environment.yaml`.

We recommend using **mamba/Miniforge** to create the environment.

### 1. Clone the repository

```bash
git clone https://github.com/Quim98/IL10_model_publication.git
cd IL10_model_publication
```

### 2. Download the Zenodo dataset

Download the archive associated with:

**https://doi.org/10.5281/zenodo.22257307**

Extract its contents directly into the repository root. After extraction, the repository should contain at least:


### 3. Create and activate the environment

```bash
mamba env create -f environment.yaml
mamba activate IL10_model
```

### 4. Run the notebooks

```bash
jupyter lab
```

---

## Main scripts and computational workflow

### RNA-to-protein abundance estimation

Cell-type-specific IL-10RA, IL-10RB, STAT1, and STAT3 protein copy numbers are estimated from RNA-seq expression using four gene-specific linear models. These estimates provide initial conditions for the mechanistic IL-10 signaling model when direct protein measurements are unavailable.

- Analysis/plots: `Plots_linear_models.ipynb`
- Inputs: `data/expression/`, `data/lin_models/`
- Main manuscript output: **Figure 2A**
- Supplementary outputs: **Appendix Figures S1 and S2**

### Mechanistic IL-10 signaling models

The repository implements the three mechanistic hypotheses evaluated in the manuscript:

1. **Baseline model** — STAT1 and STAT3 are activated only while the complete IL-10/IL-10RA/IL-10RB signaling complex is assembled.
2. **Receptor Memory model** — phosphorylated IL-10RA/JAK1 states can transiently retain STAT3 signaling after receptor dissociation, whereas STAT1 remains dependent on the intact signaling complex.
3. **Kinetic Proofreading model** — STAT1 recruitment includes an additional adaptor-dependent step, while STAT3 recruitment is direct.

The full ODE implementations and shared simulation utilities are contained in `src/`. Simplified steady-state representations used to accelerate model selection rely on files in `SS_search/`.

### Sloppiness / parameter-sensitivity analysis

Before ABC-SMC inference, a sloppiness analysis is used to identify parameter combinations that strongly affect model performance and to restrict inference to sufficiently informative parameters.

- Computation: `slopiness_analysis.py`
- Visualization: `Plots_slopiness_analysis.ipynb`
- Manuscript output: **Figure EV2**

### ABC-SMC model selection and parameter inference

Approximate Bayesian Computation–Sequential Monte Carlo (ABC-SMC) is used to compare the Baseline, Receptor Memory, and Kinetic Proofreading models and infer kinetic parameters around literature-derived priors.

Main inference scripts:

- `pyabc_inference_WT_ODE.py` — inference for WT IL-10 using the full ODE formulation.
- `pyabc_inference_mut_ODE.py` — inference/simulation workflow for IL-10 variants with altered receptor-binding kinetics.
- `pyabc_inference_mut_ODE_IL10M.py` — corresponding workflow for monomeric IL-10 variants.

Results are stored in / supplied through `results/ABC_SMC/`.

- Visualization: `Plots_ABC_SMC.ipynb`
- Main manuscript outputs: **Figure 2B-F**
- Supplementary output: **Appendix Figure S3**

### In-vitro dose-response simulations

The fitted models are simulated across IL-10 concentrations, cell types, and IL-10 variants to generate pSTAT1 and pSTAT3 dose-response curves and derived EC50/amplitude distributions **(Figures 3 and EV3, Appendix Figures S4, S6, S9-S12)**.

- `sim_invitro_fit_eIC.py` — simulations using the simplified/steady-state framework.
- `sim_invitro_fit_eIC_ODE.py` — simulations using the full ODE model.
- `sim_invitro_IL10M_eIC.py` — simulations using the full ODE model (monomeric IL-10).
- `sim_invitro_scIL10_eIC.py` — simulations using the full ODE model (single-chain IL-10).
- `model_perturbations/` — alternative model formulations and parameter perturbations used to test mechanistic predictions and high-dose signaling behavior.

The associated plotting notebook is `Plots_dose_response.ipynb`.

### Model-performance analyses

`Mech_models_performance.ipynb` compares model errors across training/test data and evaluates whether performance is systematically biased by cell type or IL-10 variant.

This notebook complements the dose-response plots by summarizing EC50 and amplitude prediction errors and produces the model-comparison panels in **Figure EV3F-G** together with the statistical/model-performance panels in **Appendix Figures S5 and S7**. This notebook also compares the performance in EC50 and Amplitude of the simplified linear framework steady-state solutions vs full ODE simulations **(Appendix Table S3)**.

`Null_model_IL10.ipynb` our mechanistic Receptor Memory IL-10 signaling model and other regression-based model alternatives (Appendix Table S2).

### Downstream gene-regulation analyses

`Plots_genome_regulation.ipynb` links model-simulated STAT activation to transcriptional responses using the Immune Dictionary and independent bulk RNA-seq datasets. It includes pSTAT-RNA correlations, pSTAT-correlated DEG identification, over-representation analyses, gene-specific pSTAT threshold fitting, transcriptional-program clustering, and validation with engineered IL-10 variants **(Figures 4 and EV4 and Appendix Figures S14, S15, S17-S24)**.

Inputs are primarily under `data/signaling/`, with model-simulated cell-type-specific STAT outputs under `results/immune_dict/`.

---

## Data availability

The datasets and code associated with this study are distributed across the following resources:

- **Code:** https://github.com/Quim98/IL10_model_publication
- **Processed experimental data, model initial conditions, inference outputs, and simulations:** Zenodo, https://doi.org/10.5281/zenodo.22257307
- **Raw quantitative proteomics data:** PRIDE, accession **PXD076154**
- External published datasets used in individual analyses are described and cited in the manuscript and within the corresponding analysis notebooks.

---

## Issues

Please report reproducibility problems, missing files, or code-related questions through the GitHub issue tracker:

https://github.com/Quim98/IL10_model_publication/issues

When reporting an issue, please include the operating system, environment/package information, script or notebook being run, and the complete error message where possible.

---

## Authors

- [Quim Martí Baena](https://orcid.org/0000-0002-6037-2979)
- [Carolina Segura-Morales](https://orcid.org/0000-0002-1746-6865)
- [Jordi García-Ojalvo](https://orcid.org/0000-0002-3716-7520)
- [Luis Serrano](https://orcid.org/0000-0002-5276-1392)

---

## Citation

If you use this code or dataset, please cite:

> Martí-Baena Q, Segura-Morales C, García-Ojalvo J, Serrano L. **Mechanistic pathway modeling reveals how IL-10 generates pleiotropic immune responses.** Preprint (2026). https://doi.org/10.64898/2026.06.02.729479
