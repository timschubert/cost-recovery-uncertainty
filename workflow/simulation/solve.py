import logging
import random
import numpy as np
import pypsa
from helpers import set_scenario_config

logger = logging.getLogger(__name__)

# Fix random seed for reproducibility of solver retries
random.seed(123)


def set_snapshots(
    n,
    number_years=False,
    random_years=False,
    fixed_year=False,
    exclude_years=None,
):
    """
    Restrict network snapshots to a subset of years.

    Parameters
    ----------
    n : pypsa.Network
        The network whose snapshots will be modified.
    number_years : int or bool, optional
        Number of years to keep. If False, all years are kept.
    random_years : bool, optional
        If True, randomly select years instead of taking the last `number_years`.
    fixed_year : int or bool, optional
        If provided, keep only this year (overrides other settings).
    exclude_years : list[int], optional
        Years to exclude from selection.

    Notes
    -----
    - If `fixed_year` is given, only that year is kept.
    - If `number_years` is False, all snapshots are kept.
    - If fewer allowed years exist than requested, raises AssertionError.
    """
    if exclude_years is None:
        exclude_years = []

    if fixed_year:
        logger.info("Fixed year %s specified. Clipping snapshots.", fixed_year)
        n.snapshots = n.snapshots[n.snapshots.year == fixed_year]
        return

    if not number_years:
        logger.info("No subset of years selected. Keep all snapshots.")
        return

    all_years = set(n.snapshots.year.unique())
    allowed_years = list(all_years.difference(exclude_years))

    assert len(allowed_years) >= number_years, (
        f"Requested {number_years} years, but only {len(allowed_years)} allowed."
    )

    if random_years:
        random.shuffle(allowed_years)
    years = allowed_years[-number_years:]

    logger.info("Clipping snapshots to years: %s", years)
    n.snapshots = n.snapshots[n.snapshots.year.isin(years)]


def solve_network(n, config, attempt=1):
    """
    Solve the network optimization problem with robust solver settings.

    Parameters
    ----------
    n : pypsa.Network
        The network to solve.
    config : dict
        Scenario configuration dictionary.
    attempt : int, optional
        Retry number (1 = first try). Later attempts may use more
        numerically robust solver settings.

    Raises
    ------
    RuntimeError
        If the solver reports infeasibility, unboundedness, or error.
    """
    solver_name = config["solver"]["name"]
    solver_profile = config["solver"]["options"]
    solver_options = config["solver_options"][solver_profile].copy()

    # Special handling for Gurobi solver
    if solver_name == "gurobi":
        logging.getLogger("gurobipy").setLevel(logging.CRITICAL)
        solver_options["threads"] = config["solver"]["threads"]

    # Retry strategy: switch to more numeric-stable Gurobi profile
    if attempt > 1 and solver_name == "gurobi":
        numeric_profile = "gurobi-numeric"
        logger.info("Retry #%d with %s solver settings.", attempt, numeric_profile)
        solver_options = config["solver_options"][numeric_profile].copy()
        solver_options["threads"] = config["solver"]["threads"]
        solver_options["NumericFocus"] = min(2, max(attempt - 1, 1))
        solver_options["Seed"] = np.random.randint(1, 999)

    # Run optimization
    status, condition = n.optimize(
        solver_name=solver_name,
        solver_options=solver_options,
        assign_all_duals=True,
    )

    if status != "ok":
        logger.warning("Solver finished with status '%s' and condition '%s'", status, condition)

    if condition in ["infeasible", "suboptimal", "unbounded", "error"]:
        raise RuntimeError(f"Optimization failed on attempt {attempt} with condition '{condition}'")


if __name__ == "__main__":
    # Mock snakemake for local debugging
    if "snakemake" not in globals():
        from helpers import mock_snakemake

        snakemake = mock_snakemake("solve", lt="inelastic+true")

    # Apply scenario config
    set_scenario_config(snakemake.config, snakemake.wildcards)

    # Load network
    n = pypsa.Network(snakemake.input.network)

    # Restrict snapshots as configured
    set_snapshots(
        n,
        snakemake.config["number_years"],
        snakemake.config["random_years"],
        snakemake.config["fixed_year"],
    )

    # Solve network (with retry if needed)
    solve_network(n, snakemake.config, snakemake.resources.attempt)

    # Store config metadata into network for traceability
    n.meta = snakemake.config

    # Export solved network and statistics
    export_kwargs = snakemake.config["export_to_netcdf"]
    n.export_to_netcdf(snakemake.output.network, **export_kwargs)
    n.statistics().to_csv(snakemake.output.statistics)
