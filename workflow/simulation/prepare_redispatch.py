import pypsa
import pandas as pd
from pypsa.descriptors import get_switchable_as_dense as as_dense
from helpers import set_scenario_config


def prepare_redispatch(n_inp_prep, n_inp_solved, n_capopt):
    # Load networks
    n_inp_prep = pypsa.Network(n_inp_prep)
    n_inp_solved = pypsa.Network(n_inp_solved)
    n_capopt = pypsa.Network(n_capopt)

    # Base network: start from prepared, fix lines from capopt
    n = n_inp_prep.copy()
    n.lines = n_capopt.lines
    n.lines["s_nom"] = n.lines["s_nom_opt"]
    n.lines["s_nom_extendable"] = False

    # Fix generator & storage capacities from solved dispatch
    n.generators.p_nom = n_inp_solved.generators.p_nom_opt
    n.storage_units.p_nom = n_inp_solved.storage_units.p_nom_opt
    n.generators["p_nom_extendable"] = False
    n.storage_units["p_nom_extendable"] = False

    # Fix generator dispatch
    p_gen_rel = (n_inp_solved.generators_t.p / n_inp_solved.generators.p_nom_opt).fillna(0)
    n.generators_t.p_min_pu = p_gen_rel
    n.generators_t.p_max_pu = p_gen_rel

    # Replace storage units with loads (PyPSA bug workaround)
    for su_name in n.storage_units.index:
        n.add("Load",
              name=f"{su_name}_storage",
              bus=n.storage_units.at[su_name, "bus"],
              carrier=n.storage_units.at[su_name, "carrier"],
              p_set=n_inp_solved.storage_units_t.p[su_name],
              sign=1)
        n.remove("StorageUnit", su_name)

    # Create up/down redispatch generators
    base_gens = n.generators[~n.generators.index.str.contains("Redispatch|load-shedding")]
    nonzero_gens = base_gens[base_gens.p_nom != 0]

    g_up = nonzero_gens.copy()
    g_down = nonzero_gens.copy()
    g_up.index = g_up.index + " ramp up"
    g_down.index = g_down.index + " ramp down"

    # Physical redispatch limits
    up = (as_dense(n_inp_solved, "Generator", "p_max_pu") * n_inp_solved.generators.p_nom
          - n_inp_solved.generators_t.p).clip(lower=0) / n_inp_solved.generators.p_nom
    down = -n_inp_solved.generators_t.p / n_inp_solved.generators.p_nom

    up = up.loc[:, nonzero_gens.index].rename(columns=lambda x: x + " ramp up")
    down = down.loc[:, nonzero_gens.index].rename(columns=lambda x: x + " ramp down")

    # Add generators and assign time-dependent parameters
    for gen_name, gen_data in g_up.drop(columns="p_max_pu").iterrows():
        n.add("Generator", gen_name, **gen_data.to_dict())
    for gen_name, gen_data in g_down.drop(columns=["p_max_pu", "p_min_pu"]).iterrows():
        n.add("Generator", gen_name, p_max_pu=0, **gen_data.to_dict())

    n.generators_t.setdefault("p_max_pu", pd.DataFrame(index=n.snapshots))
    n.generators_t.setdefault("p_min_pu", pd.DataFrame(index=n.snapshots))
    n.generators_t.p_max_pu = pd.concat([n.generators_t.p_max_pu, up], axis=1)
    n.generators_t.p_min_pu = pd.concat([n.generators_t.p_min_pu, down], axis=1)

    # Add reserve OCGTs
    for bus in n.buses.index:
        n.add("Generator", f"Redispatch OCGT Reserve {bus}",
              bus=bus, p_nom_extendable=True,
              carrier="OCGT", marginal_cost=64.683952,
              capital_cost=47718.671875, efficiency=0.410004)

    # Remove CO2 constraint if exists
    n.global_constraints.drop("CO2Limit", errors="ignore", inplace=True)

    return n


if __name__ == "__main__":
    snakemake = globals().get("snakemake")
    set_scenario_config(snakemake.config, snakemake.wildcards)

    n_inp_prep = snakemake.input.prepared_dispatch
    n_inp_solved = snakemake.input.solved_dispatch
    n_inp_capopt_fk = snakemake.input.solved_capopt_fk

    n_prep_redispatch = prepare_redispatch(n_inp_prep, n_inp_solved, n_inp_capopt_fk)
    n_prep_redispatch.export_to_netcdf(snakemake.output.network_prepared,
                                       **snakemake.config["export_to_netcdf"])
