import logging
import warnings
import pandas as pd
import numpy as np
import pypsa
from helpers import set_scenario_config
import cluster_network

logger = logging.getLogger(__name__)

# Suppress specific xarray DeprecationWarnings
warnings.filterwarnings(
    "ignore",
    message="Deleting a single level of a MultiIndex is deprecated.*",
    category=DeprecationWarning,
)


def cluster(network, snapshots, number_of_clusters, config):
    """
    Cluster the network for computational performance.

    Parameters
    ----------
    network : pypsa.Network
        Original network.
    snapshots : pd.DatetimeIndex
        Snapshots to retain in the clustered network.
    number_of_clusters : int
        Number of clusters for the network.
    config : dict
        Scenario configuration.

    Returns
    -------
    pypsa.Network
        Clustered network.
    """
    cluster_network.cluster_network(
        network,
        snapshots,
        number_of_clusters,
        config,
        input_directory="data/input_networks/",
        output_directory="data/input_networks/",
    )

    clustered_net = pypsa.Network()
    clustered_net.import_from_netcdf(
        f"data/input_networks/base_s_{number_of_clusters}_elec_.nc"
    )

    return clustered_net


def cluster_generators_per_bus(net):
    """
    Aggregate generators and storage units by bus and carrier.

    Behavior:
    - One generator per bus and carrier is kept.
    - p_nom_min is set to 0.
    - p_nom_max is the sum of all original units of that carrier at the bus.
    - Time series (p_max_pu) is transferred from the first unit if available.

    Parameters
    ----------
    net : pypsa.Network

    Returns
    -------
    pypsa.Network
    """
    gen_carriers = net.generators.carrier.unique()
    storage_carriers = net.storage_units.carrier.unique()

    # Aggregate generators
    for bus in net.buses.index:
        for carrier in gen_carriers:
            gens = net.generators[(net.generators.bus == bus) & (net.generators.carrier == carrier)]
            if not gens.empty:
                p_nom_max_sum = gens['p_nom_max'].sum()
                gen = gens.iloc[0].copy()
                gen['p_nom_min'] = 0
                gen['p_nom_max'] = p_nom_max_sum

                dropgens = gens.index
                net.generators = net.generators.drop(dropgens)
                net.add("Generator", f"{bus}_{carrier}", **gen.to_dict())

                # Update p_max_pu time series
                if 'p_max_pu' in net.generators_t and gen.name in net.generators_t.p_max_pu.columns:
                    p_max_pu = net.generators_t.p_max_pu[gen.name]
                    net.generators_t.p_max_pu.drop(dropgens, inplace=True, axis=1)
                    net.generators_t.p_max_pu[f"{bus}_{carrier}"] = p_max_pu

        # Aggregate storage units
        for carrier in storage_carriers:
            stores = net.storage_units[(net.storage_units.bus == bus) & (net.storage_units.carrier == carrier)]
            if not stores.empty:
                p_nom_max_sum = stores['p_nom_max'].sum()
                sto = stores.iloc[0].copy()
                sto['p_nom_min'] = 0
                sto['p_nom_max'] = p_nom_max_sum

                dropstorage = stores.index
                net.storage_units = net.storage_units.drop(dropstorage)
                net.add("StorageUnit", f"{bus}_{carrier}", **sto.to_dict())

    return net


def cluster_loads_per_bus(net):
    """
    Aggregate all loads at each bus.

    Behavior:
    - Single load per bus.
    - p_set time series is summed across all loads at the bus.

    Parameters
    ----------
    net : pypsa.Network

    Returns
    -------
    pypsa.Network
    """
    for bus in net.buses.index:
        bus_loads = net.loads[net.loads.bus == bus]
        if bus_loads.empty:
            continue

        # Sum p_set across loads
        p_set_sum = net.loads_t.p_set[bus_loads.index].sum(axis=1)
        load = bus_loads.iloc[0].copy()

        net.loads = net.loads.drop(bus_loads.index)
        net.loads_t.p_set = net.loads_t.p_set.drop(bus_loads.index, axis=1)

        net.add("Load", f"{bus}_load", **load.to_dict())
        net.loads_t.p_set[f"{bus}_load"] = p_set_sum

    return net


def set_transmission_limit(n, factor, Nyears=1):
    """
    Set or constrain transmission expansion limits.

    Parameters
    ----------
    n : pypsa.Network
    factor : str or float
        Expansion factor or "opt" to allow optimization.
    Nyears : int
        Number of years for scaling (default=1).

    Returns
    -------
    pypsa.Network
    """
    links_dc_b = n.links.carrier == "DC" if not n.links.empty else pd.Series(dtype=bool)

    # Compute line nominal capacities if not explicitly defined
    _lines_s_nom = (
        np.sqrt(3)
        * n.lines.type.map(n.line_types.i_nom)
        * n.lines.num_parallel
        * n.lines.bus0.map(n.buses.v_nom)
    )
    lines_s_nom = n.lines.s_nom.where(n.lines.type == "", _lines_s_nom)

    col = "length"
    ref = (
        lines_s_nom @ n.lines[col]
        + n.links.loc[links_dc_b, "p_nom"] @ n.links.loc[links_dc_b, col]
    )

    # If optimization or factor > 1, make lines extendable
    if factor == "opt" or float(factor) > 1.0:
        n.lines["s_nom_min"] = lines_s_nom
        n.lines["s_nom_extendable"] = True
        n.links.loc[links_dc_b, "p_nom_min"] = n.links.loc[links_dc_b, "p_nom"]
        n.links.loc[links_dc_b, "p_nom_extendable"] = True

    # Add global constraint for line expansion if not optimization
    if factor != "opt":
        con_type = "volume_expansion"
        rhs = float(factor) * ref
        n.add(
            "GlobalConstraint",
            "lv_limit",
            type=f"transmission_{con_type}_limit",
            sense="<=",
            constant=rhs,
            carrier_attribute="AC, DC",
        )

    return n


if __name__ == "__main__":
    if "snakemake" not in globals():
        from helpers import mock_snakemake
        from pathlib import Path

        snakemake = mock_snakemake(
            "prepare_network",
            lt="number_years+1-elastic+true-elastic_intercept+200",
            configfiles=[Path("../config/config.yaml")],
        )

    set_scenario_config(snakemake.config, snakemake.wildcards)

    # Import network
    network = pypsa.Network(snakemake.input.network)
    network.lines["dc"] = network.lines["dc"].fillna(0.0)

    # Define snapshots
    snapshots = pd.date_range(
        snakemake.config["snapshots"]["start"],
        snakemake.config["snapshots"]["end"],
        freq=snakemake.config["snapshots"]["freq"],
    )
    network.set_snapshots(snapshots)

    # Optional clustering
    if snakemake.config["input_network"]["no_of_clusters"]:
        network = cluster(network, snapshots, snakemake.config["input_network"]["no_of_clusters"], snakemake.config)

    # Aggregate generators and loads for performance
    network = cluster_generators_per_bus(network)
    network = cluster_loads_per_bus(network)

    # Remove links under construction
    if not network.links.empty and "under_construction" in network.links.columns:
        for link in network.links[network.links["under_construction"] == True].index:
            network.remove("Link", link)

    # Reset capacities for greenfield scenario
    network.generators["p_nom"] = 0.0
    network.generators["p_nom_extendable"] = True
    network.storage_units["p_nom"] = 0.0
    network.storage_units["p_nom_extendable"] = True

    # Add zone attribute
    network.buses["zone"] = "Nodal"

    # Assign unique carriers for loads
    for load in network.loads.index:
        carrier = f"load_{network.loads.at[load, 'bus']}"
        network.add("Carrier", carrier)
        network.loads.at[load, "carrier"] = carrier

    # Add VOLL load shedding generators
    for bus in network.buses.index:
        network.add(
            "Generator",
            f"load-shedding-VOLL_{bus}",
            bus=bus,
            p_nom=float("inf"),
            marginal_cost=snakemake.config["voll_price"],
            carrier="AC",
        )

    # Apply CO2 emission cap
    if snakemake.config["co2_cap"] != False:
        cap = snakemake.config["co2_cap"] * len(snapshots) / 8760
        network.add(
            "GlobalConstraint",
            "CO2Limit",
            type="primary_energy",
            carrier_attribute="co2_emissions",
            sense="<=",
            constant=cap,
        )

    # Adjust line expansion limit
    if "lv_limit" in network.global_constraints.index:
        network.remove("GlobalConstraint", "lv_limit")
    network = set_transmission_limit(network, snakemake.config["line_expansion_limit"])

    # Export processed network
    export_kwargs = snakemake.config["export_to_netcdf"]
    network.export_to_netcdf(snakemake.output.network, **export_kwargs)
