import pypsa
from helpers import set_scenario_config


def scale_line_capacities(n_prep, n_solved, config):
    """
    Scale line capacities of a prepared PyPSA network based on optimization results.

    Parameters
    ----------
    n_prep : str | pypsa.Network
        Prepared network or path to it.
    n_solved : str | pypsa.Network
        Solved network or path to it.
    config : dict
        Must contain:
            config["gen_knowledge_uncertainty"]["line_capacity_fac"]

    Returns
    -------
    pypsa.Network
        Updated prepared network with scaled line capacities.
    """
    # Load networks
    n_prep = pypsa.Network(n_prep) if isinstance(n_prep, str) else n_prep
    n_solved = pypsa.Network(n_solved) if isinstance(n_solved, str) else n_solved
    scaling_fac = config["gen_knowledge_uncertainty"]["line_capacity_fac"]

    # Scale solved line capacities
    lines = n_solved.lines.copy()
    lines["s_nom_opt"] *= scaling_fac
    lines["s_nom"] = lines["s_nom_opt"]
    lines["s_nom_extendable"] = False

    n_prep.lines = lines

    # Drop global constraint if present
    if "lv_limit" in n_prep.global_constraints.index:
        n_prep.global_constraints.drop("lv_limit", inplace=True)

    return n_prep


def adjust_generator_knowledge(n_prep, n_solved, config):
    """
    Fix generator and storage capacities for technologies with full knowledge.

    Parameters
    ----------
    n_prep : str | pypsa.Network
        Prepared network or path to it.
    n_solved : str | pypsa.Network
        Solved network or path to it.
    config : dict
        Must contain:
            config["gen_knowledge_uncertainty"]["gen_techs"]
            config["gen_knowledge_uncertainty"]["storage_techs"]

    Returns
    -------
    pypsa.Network
        Prepared network with fixed capacities for certain techs.
    """
    # Load networks
    n_prep = pypsa.Network(n_prep) if isinstance(n_prep, str) else n_prep
    n_solved = pypsa.Network(n_solved) if isinstance(n_solved, str) else n_solved

    uncertain_gen_techs = config["gen_knowledge_uncertainty"]["gen_techs"]
    uncertain_storage_techs = config["gen_knowledge_uncertainty"]["storage_techs"]

    # Identify certain technologies
    all_gen_techs = n_prep.generators.carrier.unique()
    all_storage_techs = n_prep.storage_units.carrier.unique()
    certain_gen_techs = [t for t in all_gen_techs if t not in uncertain_gen_techs]
    certain_storage_techs = [t for t in all_storage_techs if t not in uncertain_storage_techs]

    # Generators: fix capacity for certain techs
    generators = n_solved.generators.copy()
    mask_gen = generators.carrier.isin(certain_gen_techs) & ~generators.index.str.contains("VOLL")
    generators.loc[mask_gen, "p_nom"] = generators.loc[mask_gen, "p_nom_opt"]
    generators.loc[mask_gen, "p_nom_extendable"] = False
    n_prep.generators = generators

    # Storage: fix capacity for certain techs
    storages = n_solved.storage_units.copy()
    mask_sto = storages.carrier.isin(certain_storage_techs)
    storages.loc[mask_sto, "p_nom"] = storages.loc[mask_sto, "p_nom_opt"]
    storages.loc[mask_sto, "p_nom_extendable"] = False
    n_prep.storage_units = storages

    return n_prep


if __name__ == "__main__":
    snakemake = globals().get("snakemake")
    set_scenario_config(snakemake.config, snakemake.wildcards)

    n_prep = snakemake.input.input_prepared
    n_solved = snakemake.input.input_solved

    n_scaled = scale_line_capacities(n_prep, n_solved, snakemake.config)
    n_updated = adjust_generator_knowledge(n_scaled, n_solved, snakemake.config)

    export_kwargs = snakemake.config["export_to_netcdf"]
    n_updated.export_to_netcdf(snakemake.output.network_prepared, **export_kwargs)
