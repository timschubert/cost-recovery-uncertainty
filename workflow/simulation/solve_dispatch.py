import logging
import pypsa
from helpers import set_scenario_config
from pypsa.descriptors import nominal_attrs


def fix_optimal_capacities_from_other(n, other):
    """
    Copy optimal capacities from another solved network and fix them in `n`.

    Parameters
    ----------
    n : pypsa.Network
        The network whose capacities should be fixed.
    other : pypsa.Network
        The solved network containing optimal capacities (with *_opt fields).

    Notes
    -----
    - Applies to all extendable components in `nominal_attrs`.
    - After copying, extendable flags are set to False to freeze capacities.
    """
    for c, attr in nominal_attrs.items():
        ext_i = n.get_extendable_i(c)
        n.df(c).loc[ext_i, attr] = other.df(c).loc[ext_i, attr + "_opt"]
        n.df(c)[attr + "_extendable"] = False


if __name__ == "__main__":
    # Mock snakemake for local debugging (ignored in simulation runs)
    if "snakemake" not in globals():
        from helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_myopic_dispatch", lt="inelastic+true", st="horizon+100"
        )

    # Apply scenario-specific config settings
    set_scenario_config(snakemake.config, snakemake.wildcards)

    # Solver setup
    solver_name = snakemake.config["myopic_solver"]["name"]
    solver_profile = snakemake.config["myopic_solver"]["options"]
    solver_options = snakemake.config["solver_options"][solver_profile].copy()

    # Load prepared and solved networks
    n = pypsa.Network(snakemake.input.prepared_network)
    n_solved = pypsa.Network(snakemake.input.solved_network)

    # Fix optimal capacities from the solved run into the dispatch network
    fix_optimal_capacities_from_other(n, n_solved)

    # Storage: set cyclic condition (carryover of energy state between periods)
    n.stores.e_cyclic = snakemake.config["myopic"]["cyclic"]

    # Adjust Gurobi solver settings
    if solver_name == "gurobi":
        logging.getLogger("gurobipy").setLevel(logging.CRITICAL)
        solver_options["threads"] = snakemake.config["myopic_solver"]["threads"]

    # Solve dispatch optimization
    n.optimize(
        solver_name=solver_name,
        solver_options=solver_options,
        assign_all_duals=True,
    )

    # Export solved network and statistics
    export_kwargs = snakemake.config["export_to_netcdf"]
    n.export_to_netcdf(snakemake.output.network, **export_kwargs)
    n.statistics().to_csv(snakemake.output.statistics)
