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


---

## Clarification on Semi-Flexible Demand Implementation 

The concept and implementation of semi-flexible demand has three functional components:
1. Multi-step elasticity calculation proposed by Kladnik et al.
2. Price-dependent elasticity in log-log form proposed by Arnold
3. Piecewise-linear approximation of elasticity curves


### Multi-Step Elasticity Calculation

The concept was originally proposed in *An assessment of the effects of demand response in electricity markets* by Kladnik et al. (2013): [10.1002/etep.666](https://doi.org/10.1002/etep.666). 
It follows a three-step approach to create a new market equilibrium point considering assumptions for demand response in the electricity system. The approach was adapted to the issue at hand to translate fixed, historic demand time series into a representation of demand elasticity.

- Step 1: Use fixed, historic demand time series as an input assumption for the model and simulate to derive hourly demand-price reference points. This step is conducted in via *solve.py* in the rule *solve_limited_knowledge*.
- Step 2: Based on these demand-price pairs create a mathematical formulation for demand elasticity. This is done in the rule *add_semi_flexible_demand* via the respective script. The exact mathematical approach is described in 2. and 3. of this clarification.
- Step 3: Re-run the simulation with demand elasticity formulation for each hour to calculate new demand-price pairs considering the elasticity formulation. This is done in the rule *solve_semi_flexible* via *solve.py*.


### Log-log Price-Dependent Elasticity

For the mathematical formulation of the elasticity function itself, the publication *On the functional form of short-term electricity demand response - insights from high-price years in Germany* by F. Arnold (2023) is used: [EWI Working Paper, No 23/06](https://www.ewi.uni-koeln.de/cms/wp-content/uploads/2023/07/EWI_WP_23-06_On_the_functional_of_short-term_electricity_demand_response_F_Arnold.pdf).
It uses statistical methods to convert historical observations for the German electricity market to derive a log-log form for demand elsticity formulation. Additionally, higher elasticity values are observed for higher price ranges.

These findings are used for the elasticity formulation in this framework.
- In *config.yaml*, the elasticity ranges and elasticity values are defined:
  ```yaml
  elasticities:
  lower_bound: [0, 50, 200]          # Price breakpoints for elasticity definition (€/MWh)
  elasticity: [-0.04, -0.05, -0.06]  # Elasticity values at each breakpoint (-0.04 = 4% price elasticity)
  ```
  In this example, the first range goes from 0 to 50 €/MWh with a price elasticity of 4%, the second one from 50 to 200 €/MWh at 5%, and the last one sits above 200 €/MWh with 6%.
- These values are then referenced in *add_semi-flexible_demand* for the log-log formulation. D refers to the demand, p to the price, and k to the scaling parameter of the curve.
  ```python
  k = np.log(D) - epsilon * np.log(p)
  ```


### Piecewise-Linear (pwl) Approximation

Calculating with log-log curves would lead to quadratic formulation and therefore would be too computationally expensive. Therefore, a three-segment, piecewise-linear approximation is chosen. The approach follows *Price formation without fuel costs: the interaction of elastic demand with storage bidding* by *Tom Brown et al.* [Link](https://doi.org/10.1016/j.eneco.2025.108483.).

- Fixed price breakpoints are defined in *config.yaml*:
  ```yaml
  segments_p: # Fixed price breakpoints for piecewise linear approximation of demand elasticity function (€/MWh)
    - [800, 400]
    - [400, 200]
    - [200, 10]
  ```
- Three segments are defined between these breakpoints. In this example, the first ranges from 10 to 200 €/MWh, the second one from 200 to 400 €/MWh and the third one from 400 to 800 €/MWh
- For each of the three segments, the nominal (=Demand values / x-intercepts), the intercept (with y-axis), and the slope are calculated based on the adjusted k-value in *add_semi_flexible_demand*.

![Illustration](workflow/analysis/readme_pwl_curve.png)
