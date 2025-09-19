"""
redispatch_calculation.py

Module for redispatch-related calculations in power system simulations.

This file collects all functions used to model redispatch behavior,
including adjustments for grid-efficient curtailment and other related
processes.
"""

import pandas as pd
import numpy as np


def apply_grid_efficient_curtailment(net, zones):
    """
    Apply grid-efficient curtailment adjustments to a given network.

    In the original simulation, redispatch may incorrectly up- and down-
    dispatch plants of the same type within a single zone and hour. This
    results in inflated redispatch volumes that do not represent grid-
    efficient behavior.

    This function corrects that behavior by applying net redispatch within
    each zone, technology, and hour. Specifically:
    - Only renewable carriers are considered (OCGT excluded).
    - Net up- or down-dispatch is calculated for each snapshot.
    - Dispatch is then reallocated proportionally to the original
      contribution of generators on the relevant side (up or down).

    Parameters
    ----------
    net : pypsa.Network
        The network object containing dispatch results.
    zones : list of str
        List of zones in which to apply grid-efficient curtailment.

    Returns
    -------
    pypsa.Network
        A copy of the input network with corrected redispatch behavior.
    """
    n = net.copy()

    # Identify redispatch generators (both ramp up and ramp down)
    up_down_gens = n.generators[
        n.generators.index.str.contains("ramp up")
        | n.generators.index.str.contains("ramp down")
    ]

    for zone in zones:
        # Get all buses and generators in the zone
        buses_in_zone = n.buses[n.buses["zone"] == zone].index
        gens_in_zone = up_down_gens[
            (up_down_gens.bus.isin(buses_in_zone)) & (up_down_gens.p_nom > 0)
        ]

        # Consider only renewable carriers
        techs_in_zone = set(gens_in_zone.carrier.unique()) - {"OCGT"}

        for tech in techs_in_zone:
            tech_gens = gens_in_zone[gens_in_zone["carrier"] == tech]

            for snapshot in n.snapshots:
                p = n.generators_t.p.loc[snapshot, tech_gens.index]

                # Only adjust if both up and down-dispatch occur simultaneously
                if (p > 0).any() and (p < 0).any():
                    p_sum = p.sum()
                    p_net = p.copy()

                    if p_sum > 0:
                        # Net up-dispatch case
                        p_net[p_net < 0] = 0
                        pos_mask = p > 0
                        pos_shares = p[pos_mask] / p[pos_mask].sum()
                        p_net[pos_mask] = p_sum * pos_shares

                    elif p_sum < 0:
                        # Net down-dispatch case
                        p_net[p_net > 0] = 0
                        neg_mask = p < 0
                        neg_shares = p[neg_mask] / p[neg_mask].sum()
                        p_net[neg_mask] = p_sum * neg_shares

                    # Update network dispatch
                    n.generators_t.p.loc[snapshot, tech_gens.index] = p_net

    return n


# redispatch_calculation.py
"""
Redispatch Calculation Functions

This module contains helper functions used to calculate redispatch-related 
volumes and costs. Functions focus on the modeling of gas reserve generators 
built for redispatch flexibility.
"""

import pandas as pd


def calculate_gas_reserve_volume(network):
    """
    Calculate per-bus, per-hour dispatch of gas reserve generators built 
    for redispatch flexibility.

    Args:
        network (pypsa.Network): Network object containing generator data.

    Returns:
        pd.DataFrame: Dispatch volume of redispatch gas reserves.
                      Index = snapshots, Columns = buses, Values = MW.
    """
    gas_reserve_volume = pd.DataFrame(
        index=network.snapshots, 
        columns=network.buses.index, 
        dtype=float, 
        data=0.0
    )

    # Identify redispatch OCGT reserve plants
    gas_reserve_plants = network.generators.loc[
        network.generators.index.str.contains("Redispatch OCGT Reserve")
    ]

    for bus in network.buses.index:
        plants_at_bus = gas_reserve_plants.loc[gas_reserve_plants.bus == bus]
        if plants_at_bus.empty:
            continue

        reserve_plant = plants_at_bus.index[0]
        gas_reserve_volume.loc[:, bus] = network.generators_t.p[reserve_plant]

    return gas_reserve_volume


def calculate_gas_reserve_costs(network, network_disp, zones):
    """
    Calculate per-bus, per-hour costs for redispatch OCGT reserve generators.

    Costs include:
    - Annualized capital costs
    - Operational costs (fuel, marginal, CO₂ costs)

    Args:
        network (pypsa.Network): Network with redispatch gas reserve plants.
        network_disp (pypsa.Network): Dispatch simulation network for CO₂ price reference.
        zones (list[str]): List of bidding zones.

    Returns:
        pd.DataFrame: Redispatch gas reserve costs.
                      Index = snapshots, Columns = buses, Values = EUR.
    """
    gas_reserve_costs = pd.DataFrame(
        index=network.snapshots, 
        columns=network.buses.index, 
        dtype=float, 
        data=0.0
    )

    # Get CO₂ price and emission factor for OCGT
    co2_price = abs(network_disp.global_constraints.loc["CO2Limit"].mu)
    em_fac_ocgt = network_disp.carriers.loc["OCGT", "co2_emissions"]

    # Identify redispatch OCGT reserve plants
    gas_reserve_plants = network.generators.loc[
        network.generators.index.str.contains("Redispatch OCGT Reserve")
    ]

    for zone in zones:
        zone_buses = network.buses.loc[network.buses["zone"] == zone].index
        total_costs_zone = pd.Series(0.0, index=network.snapshots)

        for bus in zone_buses:
            plants_at_bus = gas_reserve_plants.loc[gas_reserve_plants.bus == bus]
            if plants_at_bus.empty:
                continue

            reserve_plant = plants_at_bus.index[0]
            p_nom = network.generators.loc[reserve_plant, "p_nom_opt"]

            if p_nom > 0:
                power = network.generators_t.p[reserve_plant]

                # Capital costs (annualized based on dispatch share)
                capital_cost_total = (
                    network.generators.loc[reserve_plant, "p_nom_opt"]
                    * network.generators.loc[reserve_plant, "capital_cost"]
                )
                capital_costs_ann = (
                    (power / power.sum()) * capital_cost_total if power.sum() > 0 else 0
                )

                # Operational costs (marginal + fuel + CO₂)
                operational_costs = power * (
                    network.generators.loc[reserve_plant, "marginal_cost"]
                    + em_fac_ocgt * co2_price
                )

                total_costs = capital_costs_ann + operational_costs
                total_costs_zone += total_costs

        # Distribute zone-wide redispatch costs equally across buses
        if len(zone_buses) > 0:
            for bus in zone_buses:
                gas_reserve_costs.loc[:, bus] = total_costs_zone / len(zone_buses)

    return gas_reserve_costs


def calculate_up_down_dispatch_volume(network):
    """
    Calculate up- and down-dispatch volumes for redispatch generators.

    Args:
        network (pypsa.Network): Network object containing generator data.

    Returns:
        tuple:
            - pd.DataFrame: Up-dispatch volumes [MW], index = snapshots, columns = generators.
            - pd.DataFrame: Down-dispatch volumes [MW], index = snapshots, columns = generators.
    """
    # Identify up- and down-dispatch generators
    up_gens = network.generators.loc[network.generators.index.str.contains("ramp up")]
    down_gens = network.generators.loc[network.generators.index.str.contains("ramp down")]

    # Extract time series of dispatch volumes
    up_p = network.generators_t.p[up_gens.index]
    down_p = network.generators_t.p[down_gens.index]

    return up_p, down_p


def calculate_up_down_dispatch_costs(network, network_disp, zones):
    """
    Calculate per-bus, per-hour redispatch costs for up- and down-dispatch generators.

    Costs include:
    - Up-dispatch: dispatched power × (marginal cost + CO₂ cost for OCGT).
    - Down-dispatch: compensation for lost revenues = max((LMP × power) - (power × marginal cost), 0).

    Args:
        network (pypsa.Network): Network with redispatch up/down generators.
        network_disp (pypsa.Network): Dispatch simulation network (for CO₂ price and LMPs).
        zones (list[str]): List of bidding zones.

    Returns:
        pd.DataFrame: Redispatch costs per bus and snapshot [EUR].
                      Index = snapshots, Columns = buses.
    """
    buses = network.buses.index
    snapshots = network.snapshots
    up_down_costs = pd.DataFrame(0.0, index=snapshots, columns=buses)

    # Get CO₂ price from dispatch simulation
    co2_price = abs(network_disp.global_constraints.loc["CO2Limit"].mu)
    em_fac_ocgt = network_disp.carriers.loc["OCGT", "co2_emissions"]

    # Identify up/down redispatch generators
    up_gens = network.generators.loc[network.generators.index.str.contains("ramp up")]
    down_gens = network.generators.loc[network.generators.index.str.contains("ramp down")]

    for zone in zones:
        zone_buses = network.buses.loc[network.buses["zone"] == zone].index
        total_costs_zone = pd.Series(0.0, index=snapshots)

        for bus in zone_buses:
            # --- Up-dispatch costs ---
            up_gens_at_bus = up_gens[(up_gens.bus == bus) & (up_gens.p_nom > 0)]
            if not up_gens_at_bus.empty:
                up_p = network.generators_t.p[up_gens_at_bus.index]

                # Adjust marginal cost for OCGT (include CO₂ costs)
                up_mcost = up_gens_at_bus.marginal_cost.copy()
                ocgt_mask = up_gens_at_bus.carrier == "OCGT"
                up_mcost.loc[ocgt_mask] += em_fac_ocgt * co2_price

                # Multiply dispatch × marginal cost
                up_cost = up_p.multiply(up_mcost, axis=1)
                total_costs_zone += up_cost.sum(axis=1)

            # --- Down-dispatch costs ---
            down_gens_at_bus = down_gens[(down_gens.bus == bus) & (down_gens.p_nom > 0)]
            if not down_gens_at_bus.empty:
                down_p = abs(network.generators_t.p[down_gens_at_bus.index])

                # Adjust marginal cost for OCGT
                down_mcost = down_gens_at_bus.marginal_cost.copy()
                ocgt_mask = down_gens_at_bus.carrier == "OCGT"
                down_mcost.loc[ocgt_mask] += em_fac_ocgt * co2_price

                # Locational marginal price (from zonal dispatch simulation)
                lmp = network_disp.buses_t.marginal_price[bus]

                # Compensation for lost revenues
                lmp_comp = lmp * down_p.sum(axis=1)

                # Saved marginal costs
                mc_savings = down_p.multiply(down_mcost, axis=1).sum(axis=1)

                # Down-dispatch cost = lost revenue - saved cost, clipped at >= 0
                total_costs_zone += (lmp_comp - mc_savings).clip(lower=0)

        # Distribute zone-wide redispatch costs equally over all buses in zone
        if len(zone_buses) > 0:
            for bus in zone_buses:
                up_down_costs.loc[:, bus] = total_costs_zone / len(zone_buses)

    return up_down_costs

