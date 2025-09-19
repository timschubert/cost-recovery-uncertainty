"""
Adjust network to (multi-)zonal setup.

This script updates a PyPSA network to reflect a zonal re-configuration:
- assigns each bus to a zone by point-in-polygon test
- marks lines as intra- or inter-zonal
- sets intra-zonal lines to very large capacity and small impedance
- scales inter-zonal line capacities by a factor from the config

The main function `adjust_network_setup` accepts either a path to a network
file or an already-loaded pypsa.Network object, and returns a modified Network.
The script is runnable under snakemake (uses the `snakemake` global) or standalone.
"""
import argparse
import logging
from pathlib import Path
from typing import Dict, Union

import geopandas as gpd
import pandas as pd
import pypsa
from shapely.geometry import Point
import yaml

from helpers import set_scenario_config

logger = logging.getLogger(__name__)


def adjust_network_setup(network: Union[str, pypsa.Network], config: Dict) -> pypsa.Network:
    """
    Adjust a PyPSA network for a new zonal configuration and update line properties.

    Parameters
    ----------
    network
        Either a path to a PyPSA network file (string/Path) or an existing
        pypsa.Network instance.
    config
        Configuration dictionary containing keys:
          - bidzone_uncertainty.dispatch_no_of_zones (int)
          - gen_knowledge_uncertainty.line_capacity_fac (float)

    Returns
    -------
    pypsa.Network
        Modified network (in-memory). Caller is responsible for saving/exporting.
    """
    # Load network if a path was given
    if isinstance(network, (str, Path)):
        n = pypsa.Network(network)
    elif isinstance(network, pypsa.Network):
        n = network
    else:
        raise TypeError("`network` must be a path or a pypsa.Network instance.")

    # GeoJSON path for zonal split geometry
    zones_file = Path(
        f"data/network_splits/DE{config['bidzone_uncertainty']['dispatch_no_of_zones']}.geojson"
    )

    if not zones_file.exists():
        logger.error("Zonal GeoJSON not found: %s", zones_file)
        raise FileNotFoundError(f"Zonal GeoJSON not found: {zones_file}")

    zones_gdf = gpd.read_file(zones_file)

    # Map each bus to a zone id using point-in-polygon
    bus_zone_map = {}
    for bus_idx, bus in n.buses.iterrows():
        point = Point(bus.x, bus.y)
        matched = zones_gdf[zones_gdf.geometry.contains(point)]
        if not matched.empty:
            bus_zone_map[bus_idx] = matched.iloc[0]["id"]
        else:
            logger.warning("Bus %s not assigned to any zone (coords: %s, %s)", bus_idx, bus.x, bus.y)

    # Create zone series aligned to network buses index
    zone_series = pd.Series(bus_zone_map, name="zone").reindex(n.buses.index)
    n.buses["zone"] = zone_series

    # Determine inter-zonal boolean for each line
    # Use reindexing to ensure alignment (handles any index mismatches)
    bus0_zones = n.buses["zone"].reindex(n.lines["bus0"]).values
    bus1_zones = n.buses["zone"].reindex(n.lines["bus1"]).values
    inter_zonal = bus0_zones != bus1_zones
    n.lines["Inter_Zonal"] = inter_zonal

    # Apply changes to intra-zonal lines
    intra_mask = ~n.lines["Inter_Zonal"].astype(bool)
    intra_idx = n.lines.loc[intra_mask].index
    HIGH_CAP = 1e8
    LOW_R = 0.001
    if not intra_idx.empty:
        n.lines.loc[intra_idx, "s_nom"] = HIGH_CAP
        n.lines.loc[intra_idx, "s_nom_opt"] = HIGH_CAP
        n.lines.loc[intra_idx, "r"] = LOW_R

    # Apply scaling to inter-zonal lines using config factor
    inter_idx = n.lines.loc[n.lines["Inter_Zonal"].astype(bool)].index
    fac = float(config["gen_knowledge_uncertainty"].get("line_capacity_fac", 1.0))
    if not inter_idx.empty:
        # Safely divide s_nom_opt by factor (if s_nom_opt exists)
        if "s_nom_opt" in n.lines.columns:
            n.lines.loc[inter_idx, "s_nom"] = n.lines.loc[inter_idx, "s_nom_opt"] / fac
            n.lines.loc[inter_idx, "s_nom_opt"] = n.lines.loc[inter_idx, "s_nom_opt"] / fac
        else:
            logger.warning("Column 's_nom_opt' not present on lines; skipping inter-zonal scaling.")

    logger.info(
        "Zonal adjustment complete: %d intra-zonal, %d inter-zonal lines",
        len(intra_idx),
        len(inter_idx),
    )

    return n


def _run_from_snake(snakemake) -> None:
    """
    Entry point used when executed by Snakemake (snakemake global is available).
    """
    set_scenario_config(snakemake.config, snakemake.wildcards)

    network_input_prep = snakemake.input.input_prepared
    network_input_solved = snakemake.input.input_solved

    network_prep_adj = adjust_network_setup(network_input_prep, snakemake.config)
    network_solved_adj = adjust_network_setup(network_input_solved, snakemake.config)

    export_kwargs = snakemake.config.get("export_to_netcdf", {})
    network_prep_adj.export_to_netcdf(snakemake.output.network_prepared, **export_kwargs)
    network_solved_adj.export_to_netcdf(snakemake.output.network_solved, **export_kwargs)


def _run_standalone() -> None:
    """
    CLI entry used for quick testing outside Snakemake.
    """
    parser = argparse.ArgumentParser(description="Adjust network zonal setup (standalone).")
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config file.")
    parser.add_argument("--input-prepared", required=True, help="Input prepared network (.nc)")
    parser.add_argument("--input-solved", required=True, help="Input solved network (.nc)")
    parser.add_argument("--output-prepared", required=True, help="Output prepared network (.nc)")
    parser.add_argument("--output-solved", required=True, help="Output solved network (.nc)")
    args = parser.parse_args()

    # Load config file (same structure as Snakemake config)
    with open(args.config, "r") as fh:
        cfg = yaml.safe_load(fh)

    # Optionally set scenario config if you have that helper behaviour outside snakemake
    try:
        set_scenario_config(cfg, {})  # keep consistent call; you may adjust as needed
    except Exception:
        # Not fatal for local quick tests; log and continue
        logger.debug("set_scenario_config not applied in standalone run.")

    network_prep_adj = adjust_network_setup(args.input_prepared, cfg)
    network_solved_adj = adjust_network_setup(args.input_solved, cfg)

    export_kwargs = cfg.get("export_to_netcdf", {})
    network_prep_adj.export_to_netcdf(args.output_prepared, **export_kwargs)
    network_solved_adj.export_to_netcdf(args.output_solved, **export_kwargs)


if __name__ == "__main__":
    # Basic logging config for standalone testing
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    snakemake = globals().get("snakemake")
    if snakemake:
        _run_from_snake(snakemake)
    else:
        _run_standalone()