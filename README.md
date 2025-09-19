# Cost Recovery under Uncertainty in Energy System Optimization Models
**Author:** Tim Schubert
**First Examiner**: Dr. Mirko Schäfer
**Second Examiner**: Marta Victoria 
**Submission Date:** August 4, 2025
**Abstract:** The debate around optimal electricity market design in Germany has regained momentum following the 2025 Bidding Zone Review, which observes increased economic efficiency under a split of Germany’s unified market. Assessing the impact of such structural policy uncertainty on market participants requires advanced energy system modeling. Yet, many existing models lack the technical granularity, transparency, and asset-level resolution needed to trace cause-effect relationships and assess financial outcomes for individual market participants. In particular, the cost recovery of existing generation and storage assets under market uncertainty remains critically underexplored.
To address this gap, this thesis develops a deterministic, multi-stage optimization framework to evaluate the financial viability of generation and storage assets under ex- post market uncertainty. The model combines capacity optimization with limited investor congestion awareness, elastic demand, scenario-based zonal dispatch, and a redispatch stage. To evaluate the model, it is applied to a stylized version of Germany’s electricity system, both under its current configuration and zonal splits proposed in the Bidding Zone Review.
Results show that outcomes are highly sensitive to grid assumptions. A greenfield- optimized grid enables siting based on renewable potential rather than demand proximity, resulting in moderate zonal market price divergence and limited redispatch reduction. Consequently, consumer prices rise in both zones. Generator-level cost recovery reveals that most assets benefit from the split. Solar plants, conventional generation, and storage break even or achieve modest profits post-split, while wind assets remain highly profitable.
The findings highlight the importance of focusing on generator-level outcomes when evaluating policy shifts. The proposed framework—once extended with brownfield grid assumptions and cross-border coupling—offers a robust tool for transparently assessing bidding zone reforms or alternative policy instruments such as dynamic grid fees.

---

## Overview  

This repository contains the workflow and model implementation developed as part of the Master's thesis. The framework builds on the publication *Price formation without fuel costs: the interaction of elastic demand with storage bidding* by *Tom Brown et al.* available at https://doi.org/10.1016/j.eneco.2025.108483.

The workflow of this framework can be structured into four steps:
1. **Installation** – setting up the conda environment.  
2. **Configuration** – defining global and scenario-specific parameters.  
3. **Running Simulations** – executing model runs locally or on HPC clusters.  
4. **Results Analysis** – analyzing outcomes in a jupyter notebook.  

The workflow is designed for transparency, reproducibility, and easy extension to new scenarios.

---

## Installation  

The model uses **conda** for environment management. Two environment files are provided:  

- `environment.yaml` – recommended default.  
- `environment-strict.yaml` – stricter version (may cause version conflicts).  

### Steps  

```sh
# Update conda
conda update conda

# Create environment
conda env create -f workflow/envs/environment.yaml

# Activate environment
conda activate cost-recovery-uncertainty
```

### Dependencies  

Main dependencies include:  

- [PyPSA](https://pypsa.org)  
- [Linopy](https://linopy.readthedocs.io)  
- [Snakemake](https://snakemake.readthedocs.io)  
- [Gurobi](https://www.gurobi.com) 

### Troubleshooting  

In the development of this model, two version conflicts persisted. These should no longer apply. Still, if the simulation fails, please check if the following fixes in the environment pypsa files resolve the issue.

1. **Error at `localrule solve` (PyPSA version conflict)**  
   - Navigate to `pypsa/optimization/optimize.py` in your conda environment.  
   - Replace:  
     ```python
     n.objective = m.objective_value
     ```  
     with:
     ```python
     n.objective = m.objective.value
     ```  

2. **Alternative PyPSA error (`constraints.py`)**  
   - Navigate to `pypsa/optimization/constraints.py`.  
   - Change line ~570 to:  
     ```python
     Bus=buses, fill_value=LinearExpression._fill_value
     ```  
     instead of `LinearExpression.fill_value`.  

---

## Configuration  

Model parameters are controlled via YAML config files:  

- **`config/config.yaml`**  
  Contains global parameters applied across all simulations.  

- **`config/config.<scenario>.yaml`**  
  Scenario-specific files (e.g., `config/config.REF-150.yaml`) overwrite global defaults for individual runs.  

All configuration files are stored in the `config/` folder.  

---

## Running Simulations  

All simulation-relevant files are in **`workflow/simulation/`**.  

### Local Execution  

Run a specific scenario from the root repository:  

```sh
snakemake -call --use-conda --conda-frontend conda --configfile config/config.REF-150.yaml
```

### HPC Cluster Execution  

Using a SLURM profile:  

```sh
snakemake -call --profile slurm --use-conda --conda-frontend conda --configfile config/config.REF-150.yaml
```

---

## Results Analysis  

Simulation results are analyzed in a dedicated Jupyter notebook:  

**`workflow/analysis/results_analysis.ipynb`**  

This notebook contains:  

- Core metrics extraction  
- Load and generation analysis  
- Cost recovery and drivers  
- Redispatch analysis
- Consumer price dynamics  
