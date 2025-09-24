import pypsa
import pandas as pd
import numpy as np
from bisect import bisect_right
from helpers import set_scenario_config
import logging

logger = logging.getLogger(__name__)


def compute_pwl_demand_segments(rp_new, config):
    """
    Compute piecewise linear (PWL) demand segments for a log-log
    demand-price elasticity function.

    Parameters
    ----------
    rp_new : tuple(float, float)
        Reference point (demand, price) for the current time step.
    config : dict
        Must contain:
            config["elasticities"]["lower_bound"]
            config["elasticities"]["elasticity"]
            config["elasticities"]["segments_p"]

    Returns
    -------
    list of list
        [intercepts, slopes, nominals] for each load-shedding segment.
    
    -------
    Clarifications:
    - D: Demand
    - p: Price
    - ε: Elasticity
    - k: Scaling parameter of the log-log function
    """
    epsilon_bounds_lower = config["elasticities"]["lower_bound"]
    epsilon_values = config["elasticities"]["elasticity"]
    segments_p = [(float(a), float(b)) for a, b in config["elasticities"]["segments_p"]]

    # k computation at new reference point
    D_new = max(rp_new[0], 1e-10) # Demand at new referene point
    p_new = max(rp_new[1], 1e-10) # Price at new reference point
    epsilon = epsilon_values[bisect_right(epsilon_bounds_lower, p_new) - 1] # Select elasticity based on price range
    k_new = np.log(D_new) - epsilon * np.log(p_new) # New scaling parameter

    # PWL approximation step I: Compute demand values for each segment
    segments_D = [
        (np.exp(epsilon * np.log(seg[0]) + k_new),
            np.exp(epsilon * np.log(seg[1]) + k_new))
        for seg in segments_p
    ]

    # PWL approximation step II: Slopes and nominals
    slopes = [
        (seg_p[0] - seg_p[1]) / (seg_d[1] - seg_d[0])
        for seg_d, seg_p in zip(segments_D, segments_p)
    ]
    nominals = [segments_D[0][1], segments_D[1][1] - segments_D[1][0]]

    # PWL approximation step III: Intercepts
    intercepts = []
    for i, (seg_d, seg_p) in enumerate(zip(segments_D, segments_p)):
        if i == 0:
            intercepts.append(seg_p[0] + slopes[i] * seg_d[0])
        else:
            intercepts.append(seg_p[0])

    # PWL approximation step IV: Last nominal from zero-intercept
    zero_intercept = intercepts[-1] / slopes[-1]
    nominals.append(zero_intercept)

    return [intercepts, slopes, nominals]


def apply_time_series_demand_flexibility(network_input_prep, network_input_solved, config):
    """
    Apply time-dependent demand elasticity by creating custom
    PWL load-shedding segments.

    Parameters
    ----------
    network_input_prep : str | pypsa.Network
        Base (prepared) PyPSA network or path.
    network_input_solved : str | pypsa.Network
        Solved network with hourly demand and prices or path.
    config : dict
        Must contain "elasticities" and "run" keys.

    Returns
    -------
    pypsa.Network
        Modified network with demand flexibility segments added.
    """
    # Load networks
    n_inp_prep = pypsa.Network(network_input_prep)
    n_inp_solv = pypsa.Network(network_input_solved)

    # Storage for export
    store_intercept, store_slope, store_nominal = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for load in n_inp_prep.loads.index:
        bus = n_inp_solv.loads.loc[load, "bus"]

        demand_ts = n_inp_solv.loads_t.p_set[load]
        price_ts = n_inp_solv.buses_t.marginal_price[bus]

        ts_parameters = pd.DataFrame(
            index=n_inp_solv.snapshots,
            columns=[f"{t} {i}" for t in ["Intercept", "Slope", "Nominal"] for i in range(3)],
        )

        # Compute PWL parameters for each timestep
        for snapshot in n_inp_solv.snapshots:
            demand_ref, price_ref = demand_ts[snapshot], price_ts[snapshot]
            intercepts, slopes, nominals = compute_pwl_demand_segments((demand_ref, price_ref), config)
            ts_parameters.loc[snapshot, [f"Intercept {i}" for i in range(3)]] = intercepts
            ts_parameters.loc[snapshot, [f"Slope {i}" for i in range(3)]] = slopes
            ts_parameters.loc[snapshot, [f"Nominal {i}" for i in range(3)]] = nominals

        # Collect results for export
        for i in range(3):
            store_intercept[f"{load}_Intercept_{i}"] = ts_parameters[f"Intercept {i}"]
            store_slope[f"{load}_Slope_{i}"] = ts_parameters[f"Slope {i}"]
            store_nominal[f"{load}_Nominal_{i}"] = ts_parameters[f"Nominal {i}"]

        # Update load with new p_set
        n_inp_prep.loads_t.p_set[load] = ts_parameters[[f"Nominal {i}" for i in range(3)]].sum(axis=1)

        # Collect generator parameters
        p_max_pu_data, cost_data, cost_quad_data = {}, {}, {}
        for i in range(3):
            gen_name = f"load-shedding-segment-{i}_{load}"
            n_inp_prep.add("Generator", gen_name, bus=bus, carrier=f"load_{bus}", p_nom=10000)

            cost_data[gen_name] = ts_parameters[f"Intercept {i}"] - ts_parameters[f"Slope {i}"] * ts_parameters[f"Nominal {i}"]
            cost_quad_data[gen_name] = ts_parameters[f"Slope {i}"] / 2
            p_max_pu_data[gen_name] = ts_parameters[f"Nominal {i}"].astype(float) / 1000

        # Merge into network
        n_inp_prep.generators_t.marginal_cost = pd.concat([n_inp_prep.generators_t.marginal_cost, pd.DataFrame(cost_data)], axis=1)
        n_inp_prep.generators_t.marginal_cost_quadratic = pd.concat([n_inp_prep.generators_t.marginal_cost_quadratic, pd.DataFrame(cost_quad_data)], axis=1)
        n_inp_prep.generators_t.p_max_pu = pd.concat([n_inp_prep.generators_t.p_max_pu, pd.DataFrame(p_max_pu_data)], axis=1)

    # Safely remove VOLL segments if they exist
    for bus in n_inp_prep.buses.index:
        gen_name = f"load-shedding-VOLL_{bus}"
        if gen_name in n_inp_prep.generators.index:
            n_inp_prep.remove("Generator", gen_name)

    return n_inp_prep


if __name__ == "__main__":
    if "snakemake" not in globals():
        from helpers import mock_snakemake
        from pathlib import Path
        snakemake = mock_snakemake(
            "prepare",
            lt="number_years+1-elastic+true-elastic_intercept+200",
            configfiles=[Path("../config/config.yaml")],
        )

    set_scenario_config(snakemake.config, snakemake.wildcards)
    n_prep = snakemake.input.input_prepared
    n_solved = snakemake.input.input_solved

    n_out = apply_time_series_demand_flexibility(n_prep, n_solved, snakemake.config)
    n_out.export_to_netcdf(snakemake.output.network_prepared, **snakemake.config["export_to_netcdf"])
