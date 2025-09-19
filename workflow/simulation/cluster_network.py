# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

# coding: utf-8
"""
Creates networks clustered to ``{cluster}`` number of zones with aggregated
buses and transmission corridors.

Outputs
-------

- ``data/input_networks/regions_onshore_base_s_{clusters}.geojson``: GeoJSON file of onshore regions after clustering.
- ``data/input_networks/regions_offshore_base_s_{clusters}.geojson``: GeoJSON file of offshore regions after clustering.
- ``data/input_networks/busmap_base_s_{clusters}.csv``: Mapping of buses from the original network to the clustered network.
- ``data/input_networks/linemap_base_s_{clusters}.csv``: Mapping of lines from the original network to the clustered network.
- ``data/input_networks/base_s_{clusters}.nc``: NetCDF file of the clustered network.

Description
-----------

This script clusters the electricity network into a specified number of zones based on buses and transmission corridors.
The clustering process uses various algorithms (e.g., HAC, k-means) and takes into account country and sub-network divisions.

"""

import logging
import warnings
from functools import reduce

import geopandas as gpd
import linopy
import pandas as pd
import pypsa
import xarray as xr
from pypsa.clustering.spatial import (
    busmap_by_greedy_modularity,
    busmap_by_hac,
    busmap_by_kmeans,
    get_clustering_from_busmap,
)
from scipy.sparse.csgraph import connected_components

# Ignore user warnings
warnings.filterwarnings(action="ignore", category=UserWarning)
idx = pd.IndexSlice
logger = logging.getLogger(__name__)

def normed(x):
    return (x / x.sum()).fillna(0.0)

def weighting_for_country(df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    w = normed(weights.reindex(df.index, fill_value=0))
    return (w * (100 / w.max())).clip(lower=1).astype(int)

def get_feature_data_for_hac(fn: str) -> pd.DataFrame:
    ds = xr.open_dataset(fn)
    feature_data = pd.concat([ds[var].to_pandas() for var in ds.data_vars], axis=0).fillna(0.0).T
    feature_data.columns = feature_data.columns.astype(str)
    return feature_data

def fix_country_assignment_for_hac(n: pypsa.Network) -> None:
    # Ensure that disconnected buses are reassigned to the correct country
    for country in n.buses.country.unique():
        m = n[n.buses.country == country].copy()
        _, labels = connected_components(m.adjacency_matrix(), directed=False)
        component = pd.Series(labels, index=m.buses.index)
        component_sizes = component.value_counts()

        if len(component_sizes) > 1:
            disconnected_bus = component[component == component_sizes.index[-1]].index[0]
            neighbor_bus = n.lines.query(
                "bus0 == @disconnected_bus or bus1 == @disconnected_bus"
            ).iloc[0][["bus0", "bus1"]]
            new_country = list(set(n.buses.loc[neighbor_bus].country) - {country})[0]
            n.buses.at[disconnected_bus, "country"] = new_country

def distribute_n_clusters_to_countries(
    n: pypsa.Network,
    n_clusters: int,
    cluster_weights: pd.Series,
    focus_weights: dict | None = None,
    solver_name: str = "scip",
) -> pd.Series:
    """
    Distribute the number of clusters across different countries in the network.
    """
    L = (cluster_weights.groupby([n.buses.country, n.buses.sub_network])
         .sum().pipe(normed))

    N = n.buses.groupby(["country", "sub_network"]).size()[L.index]

    assert n_clusters >= len(N) and n_clusters <= N.sum(), (
        f"Number of clusters must be {len(N)} <= n_clusters <= {N.sum()} for this selection of countries."
    )

    if isinstance(focus_weights, dict):
        total_focus = sum(list(focus_weights.values()))
        assert total_focus <= 1.0, ("The sum of focus weights must be less than or equal to 1.")
        for country, weight in focus_weights.items():
            L[country] = weight / len(L[country])

        remainder = [c not in focus_weights.keys() for c in L.index.get_level_values("country")]
        L[remainder] = L.loc[remainder].pipe(normed) * (1 - total_focus)

    m = linopy.Model()
    clusters = m.add_variables(
        lower=1, upper=N, coords=[L.index], name="n", integer=True
    )
    m.add_constraints(clusters.sum() == n_clusters, name="tot")
    m.objective = (clusters * clusters - 2 * clusters * L * n_clusters).sum()
    m.solve(solver_name=solver_name)
    return m.solution["n"].to_series().astype(int)

def busmap_for_n_clusters(
    n: pypsa.Network,
    n_clusters_c: pd.Series,
    cluster_weights: pd.Series,
    algorithm: str = "kmeans",
    features: pd.DataFrame | None = None,
    **algorithm_kwds,
) -> pd.Series:
    """
    Generate a bus mapping for the clustered network using the specified algorithm.
    """
    if algorithm == "hac" and features is None:
        raise ValueError("For HAC clustering, features must be provided.")

    if algorithm == "kmeans":
        algorithm_kwds.setdefault("n_init", 1000)
        algorithm_kwds.setdefault("max_iter", 30000)
        algorithm_kwds.setdefault("tol", 1e-6)
        algorithm_kwds.setdefault("random_state", 0)

    def busmap_for_country(x):
        prefix = x.name[0] + x.name[1] + " "
        if len(x) == 1:
            return pd.Series(prefix + "0", index=x.index)
        weight = weighting_for_country(x, cluster_weights)
        if algorithm == "kmeans":
            return prefix + busmap_by_kmeans(
                n, weight, n_clusters_c[x.name], buses_i=x.index, **algorithm_kwds
            )
        elif algorithm == "hac":
            return prefix + busmap_by_hac(
                n, n_clusters_c[x.name], buses_i=x.index, feature=features.reindex(x.index, fill_value=0.0)
            )
        elif algorithm == "modularity":
            return prefix + busmap_by_greedy_modularity(
                n, n_clusters_c[x.name], buses_i=x.index
            )
        else:
            raise ValueError(f"Algorithm must be one of 'kmeans', 'hac', or 'modularity'.")

    return (
        n.buses.groupby(["country", "sub_network"], group_keys=False)
        .apply(busmap_for_country, include_groups=False)
        .squeeze()
        .rename("busmap")
    )

def clustering_for_n_clusters(
    n: pypsa.Network,
    busmap: pd.Series,
    aggregation_strategies: dict | None = None,
) -> pypsa.clustering.spatial.Clustering:
    """
    Perform clustering based on bus mapping and aggregation strategies.
    """
    if aggregation_strategies is None:
        aggregation_strategies = dict()

    line_strategies = aggregation_strategies.get("lines", dict())
    bus_strategies = aggregation_strategies.get("buses", dict())
    bus_strategies.setdefault("substation_lv", lambda x: bool(x.sum()))
    bus_strategies.setdefault("substation_off", lambda x: bool(x.sum()))

    clustering = get_clustering_from_busmap(
        n, busmap, bus_strategies=bus_strategies, line_strategies=line_strategies
    )

    return clustering

def cluster_regions(
    busmaps: tuple | list, regions: gpd.GeoDataFrame, with_country: bool = False
) -> gpd.GeoDataFrame:
    """
    Cluster regions based on busmaps and save the results to a file and to the network.
    """
    busmap = reduce(lambda x, y: x.map(y), busmaps[1:], busmaps[0])
    columns = ["name", "country", "geometry"] if with_country else ["name", "geometry"]
    regions = regions.reindex(columns=columns).set_index("name")
    regions_c = regions.dissolve(busmap)
    regions_c.index.name = "name"
    return regions_c.reset_index()

def cluster_network(n, snapshots, n_clusters, config_params, custom_busmap_filepath=None, hac_features=None, input_directory='./inputs', output_directory='./outputs'):
    """
    Main function to perform network clustering before optimization, based on the provided parameters.
    """
    buses_prev, lines_prev, links_prev = len(n.buses), len(n.lines), len(n.links)

    # Cluster Network based on mean static load for 2013
    avg_maxium_load = n.loads_t.p_set.loc[snapshots].mean(axis=0)

    if n_clusters == len(n.buses):
        busmap = n.buses.index.to_series()
        linemap = n.lines.index.to_series()
        clustering = pypsa.clustering.spatial.Clustering(n, busmap, linemap)
    else:
        custom_busmap = config_params['enable']['custom_busmap']
        if custom_busmap:
            custom_busmap = pd.read_csv(custom_busmap_filepath, index_col=0).squeeze()
            custom_busmap.index = custom_busmap.index.astype(str)
            logger.info(f"Imported custom busmap from {custom_busmap_filepath}")
            busmap = custom_busmap
        else:
            algorithm = config_params['clustering']['cluster_network']["algorithm"]
            features = None
            if algorithm == "hac":
                features = get_feature_data_for_hac(hac_features)
                fix_country_assignment_for_hac(n)

            n.determine_network_topology()

            n_clusters_c = distribute_n_clusters_to_countries(
                n, n_clusters, avg_maxium_load,
                focus_weights=config_params['clustering']['focus_weights'],
                solver_name=config_params['solver']['name'],
            )

            busmap = busmap_for_n_clusters(
                n, n_clusters_c, cluster_weights=avg_maxium_load, algorithm=algorithm, features=features
            )

        clustering = clustering_for_n_clusters(
            n, busmap, aggregation_strategies=config_params['clustering']['aggregation_strategies']
        )

    nc = clustering.n

    for attr in ["busmap", "linemap"]:
        getattr(clustering, attr).to_csv(f'{output_directory}/{attr}_base_s_{n_clusters}.csv')

    for which in ["regions_onshore", "regions_offshore"]:
        regions = gpd.read_file(f'{input_directory}/{which}_base_s.geojson')
        clustered_regions = cluster_regions((clustering.busmap,), regions)
        clustered_regions.to_file(f'{output_directory}/{which}_base_s_{n_clusters}.geojson')

    nc.meta = dict(config_params, **dict(wildcards=n_clusters))
    nc.export_to_netcdf(f'{output_directory}/base_s_{n_clusters}_elec_.nc')

    logger.info(
        f"Clustered network:\n"
        f"Buses: {buses_prev} to {len(nc.buses)}\n"
        f"Lines: {lines_prev} to {len(nc.lines)}\n"
        f"Links: {links_prev} to {len(nc.links)}"
    )
