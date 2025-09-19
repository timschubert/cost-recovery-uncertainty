import numpy as np
import pandas as pd


def get_capacities_per_zone(n, n_ref, techs):
    """
    Compute installed capacities per zone and technology.

    Parameters
    ----------
    n : pypsa.Network
        Optimized network.
    n_ref : pypsa.Network
        Reference network with bus-to-zone mapping.
    techs : list[str]
        List of technologies (carriers) to include.

    Returns
    -------
    pd.DataFrame
        Index: technologies, Columns: zones, Values: installed capacity (MW).
    """
    zones = n_ref.buses["zone"].unique()
    total_capacities = pd.DataFrame(0.0, index=techs, columns=zones)

    gen = n.generators[["bus", "carrier", "p_nom_opt"]].copy()
    sto = n.storage_units[["bus", "carrier", "p_nom_opt"]].copy()

    bus_zone_map = n_ref.buses["zone"].to_dict()
    gen["zone"] = gen["bus"].map(bus_zone_map)
    sto["zone"] = sto["bus"].map(bus_zone_map)

    for tech in techs:
        tech_df = sto if tech == "battery" else gen
        cap_by_zone = tech_df[tech_df["carrier"] == tech].groupby("zone")["p_nom_opt"].sum()
        total_capacities.loc[tech, cap_by_zone.index] = cap_by_zone.values

    return total_capacities


def build_component_metadata(net, generation_techs=None, storage_techs=None):
    """
    Build a metadata DataFrame for all generators and storage units in `net`.

    Returns
    -------
    pd.DataFrame indexed by component name with columns:
        - bus : bus id
        - zone : bus zone (if present; otherwise NaN)
        - technology : carrier name
        - type : 'generator' or 'storage'
    """
    # Generators
    gens = net.generators.copy()
    gens = gens.assign(component=gens.index, bus=gens['bus'], technology=gens['carrier'])
    gens['zone'] = gens['bus'].map(net.buses['zone'])
    gens['type'] = 'generator'
    gen_meta = gens[['component', 'bus', 'zone', 'technology', 'type']].set_index('component')

    # StorageUnits
    stos = net.storage_units.copy()
    stos = stos.assign(component=stos.index, bus=stos['bus'], technology=stos['carrier'])
    stos['zone'] = stos['bus'].map(net.buses['zone'])
    stos['type'] = 'storage'
    sto_meta = stos[['component', 'bus', 'zone', 'technology', 'type']].set_index('component')

    meta = pd.concat([gen_meta, sto_meta], axis=0)
    return meta


def get_sorted_zone_ldc(n, n_ref=None):
    """
    Compute load duration curves (LDCs) per zone.

    Parameters
    ----------
    n : pypsa.Network
        Optimized network.
    n_ref : pypsa.Network, optional
        Reference network with zone mapping. Defaults to `n`.

    Returns
    -------
    dict[str, pd.Series]
        Keys: zones, Values: sorted load duration curve.
    """
    if n_ref is None:
        n_ref = n

    zones = n_ref.buses["zone"].unique()
    eb = n.statistics.energy_balance(aggregate_time=False)

    # collect all load time series per bus
    load_df = pd.DataFrame({
        bus: eb.xs(f"load_{bus}", axis=0, level="carrier").sum().mul(-1)
        for bus in n.buses.index if f"load_{bus}" in eb.index.get_level_values("carrier")
    })

    # ensure only buses that exist in reference network
    bus_zone = n_ref.buses["zone"]
    load_df = load_df.loc[:, load_df.columns.intersection(bus_zone.index)]

    load_duration_by_zone = {}
    for zone in zones:
        buses = bus_zone[bus_zone == zone].index.intersection(load_df.columns)
        if len(buses) == 0:
            continue
        zone_load = load_df[buses].sum(axis=1)
        sorted_ldc = zone_load.sort_values(ascending=False)
        sorted_ldc.index = np.linspace(0, 100, len(sorted_ldc), endpoint=False)
        load_duration_by_zone[zone] = sorted_ldc

    return load_duration_by_zone


def get_pdc_by_zone(n):
    """
    Compute zonal price duration curves (PDCs).

    Parameters
    ----------
    n : pypsa.Network

    Returns
    -------
    pd.DataFrame
        Index: percentile (0–100), Columns: zones, Values: zonal prices (€/MWh).
    """
    price_ts_by_zone = pd.DataFrame(index=n.snapshots, columns=n.buses.zone.unique())
    energy_balance = n.statistics.energy_balance(aggregate_time=False)

    for zone in n.buses.zone.unique():
        zone_buses = n.buses[n.buses.zone == zone].index

        # load profiles per bus
        load_ts_by_bus = {
            bus: energy_balance.xs(f"load_{bus}", axis=0, level="carrier").sum().mul(-1)
            for bus in zone_buses
        }
        load_ts_by_bus = pd.DataFrame(load_ts_by_bus)

        # weight by relative bus load
        load_sums = load_ts_by_bus.sum(axis=1)
        bus_weightings = load_ts_by_bus.div(load_sums, axis=0)

        # marginal prices per bus
        price_ts_by_bus = n.buses_t.marginal_price[zone_buses]

        # volume-weighted average price per zone
        price_ts_by_zone[zone] = (price_ts_by_bus * bus_weightings).sum(axis=1)

    # sort into price duration curve
    zonal_pdc = price_ts_by_zone.apply(lambda col: col.sort_values(ascending=False).reset_index(drop=True))
    zonal_pdc.index = np.linspace(0, 100, len(zonal_pdc), endpoint=False)

    return zonal_pdc


def get_economic_metrics(n, generation_techs, storage_techs):
    """
    Compute revenues, costs, and generation for all relevant assets.

    Parameters
    ----------
    n : pypsa.Network
    generation_techs : list[str]
    storage_techs : list[str]

    Returns
    -------
    pd.DataFrame
        Index: asset IDs, Columns: [Gross Revenue, Input Costs, Capital Costs, Total Generation].
    """
    econ_metrics = pd.DataFrame()

    marginal_prices = n.buses_t.marginal_price
    co2_price = abs(n.global_constraints.loc["CO2Limit", "mu"])

    # --- Generators ---
    if generation_techs:
        gen_data = n.generators[n.generators.carrier.isin(generation_techs)]
        active = n.generators_t.p.columns[(n.generators_t.p > 0).sum() > 0]
        gen_data = gen_data.loc[gen_data.index.intersection(active)]

        if not gen_data.empty:
            gen_power = n.generators_t.p[gen_data.index]
            gen_prices = pd.concat([marginal_prices[bus] for bus in gen_data.bus], axis=1)
            gen_prices.columns = gen_data.index

            emission_fac = n.carriers.loc[gen_data.carrier, "co2_emissions"].values
            efficiency = gen_data.efficiency.values
            marginal_costs = gen_data.marginal_cost.values + co2_price * emission_fac / efficiency

            gen_econ = pd.DataFrame({
                "Gross Revenue": (gen_power * gen_prices).sum(axis=0),
                "Input Costs": gen_power.multiply(marginal_costs, axis=1).sum(axis=0),
                "Capital Costs": gen_data.capital_cost.values * gen_data.p_nom_opt.values,
                "Total Generation": gen_power.sum(axis=0),
            }, index=gen_data.index)
            econ_metrics = pd.concat([econ_metrics, gen_econ])

    # --- Storage ---
    if storage_techs:
        sto_data = n.storage_units[n.storage_units.carrier.isin(storage_techs)]
        active = n.storage_units_t.p.columns[(n.storage_units_t.p > 0).sum() > 0]
        sto_data = sto_data.loc[sto_data.index.intersection(active)]

        if not sto_data.empty:
            sto_power = n.storage_units_t.p[sto_data.index]
            sto_prices = pd.concat([marginal_prices[bus] for bus in sto_data.bus], axis=1)
            sto_prices.columns = sto_data.index

            sto_econ = pd.DataFrame({
                "Gross Revenue": (sto_power.clip(lower=0) * sto_prices).sum(axis=0),
                "Input Costs": sto_power.clip(upper=0).abs().multiply(sto_prices, axis=1).sum(axis=0),
                "Capital Costs": sto_data.capital_cost.values * sto_data.p_nom_opt.values,
                "Total Generation": sto_power.clip(lower=0).sum(axis=0),
            }, index=sto_data.index)
            econ_metrics = pd.concat([econ_metrics, sto_econ])

    return econ_metrics


def calculate_cost_recovery(n, generation_techs, storage_techs, method="gross"):
    """
    Calculate cost recovery factor (CRF) for all relevant assets.

    Parameters
    ----------
    n : pypsa.Network
    generation_techs : list[str]
    storage_techs : list[str]
    method : {"gross","net"}
        - gross: R / (IC + CC)
        - net: (R - IC) / CC

    Returns
    -------
    pd.Series
        Index: asset IDs, Values: cost recovery factors.
    """
    metrics = get_economic_metrics(n, generation_techs, storage_techs)
    metrics = metrics.loc[(metrics["Capital Costs"] + metrics["Input Costs"]) > 0]

    if method == "gross":
        return metrics["Gross Revenue"] / (metrics["Input Costs"] + metrics["Capital Costs"])
    elif method == "net":
        return (metrics["Gross Revenue"] - metrics["Input Costs"]) / metrics["Capital Costs"]
    else:
        raise ValueError("Invalid method. Choose 'gross' or 'net'.")


def get_cost_recovery_ts(n, generation_techs, storage_techs):
    """
    Time series of cost recovery factor for generators and storage.

    Parameters
    ----------
    n : pypsa.Network
    generation_techs : list[str]
    storage_techs : list[str]

    Returns
    -------
    pd.DataFrame
        Index: snapshots, Columns: asset IDs, Values: CRF per hour.
    """
    marginal_prices = n.buses_t.marginal_price
    crf = pd.DataFrame(index=n.snapshots)

    # Generators
    if generation_techs:
        gen_data = n.generators[n.generators.carrier.isin(generation_techs)]
        active = n.generators_t.p.columns[(n.generators_t.p > 0).sum() > 0]
        gen_data = gen_data.loc[gen_data.index.intersection(active)]

        if not gen_data.empty:
            gen_power = n.generators_t.p[gen_data.index]
            gen_prices = pd.concat([marginal_prices[bus] for bus in gen_data.bus], axis=1)
            gen_prices.columns = gen_data.index
            marginal_costs = gen_data.marginal_cost.values

            gen_gross = gen_power * gen_prices
            gen_input = gen_power.multiply(marginal_costs, axis=1)
            capex = gen_data.capital_cost.values * gen_data.p_nom_opt.values
            capex_distr = np.tile(capex / len(n.snapshots), (len(n.snapshots), 1))

            gen_crf = gen_gross / (gen_input + capex_distr)
            crf = pd.concat([crf, gen_crf], axis=1)

    # Storage
    if storage_techs:
        sto_data = n.storage_units[n.storage_units.carrier.isin(storage_techs)]
        active = n.storage_units_t.p.columns[(n.storage_units_t.p > 0).sum() > 0]
        sto_data = sto_data.loc[sto_data.index.intersection(active)]

        if not sto_data.empty:
            sto_power = n.storage_units_t.p[sto_data.index]
            sto_prices = pd.concat([marginal_prices[bus] for bus in sto_data.bus], axis=1)
            sto_prices.columns = sto_data.index

            sto_gross = sto_power.clip(lower=0) * sto_prices
            sto_input = sto_power.clip(upper=0).abs() * sto_prices
            capex = sto_data.capital_cost.values * sto_data.p_nom_opt.values
            capex_distr = np.tile(capex / len(n.snapshots), (len(n.snapshots), 1))

            sto_crf = sto_gross / (sto_input + capex_distr)
            crf = pd.concat([crf, sto_crf], axis=1)

    return crf


def compute_crf_by_zone(net, econ_metrics, generation_techs, storage_techs):
    """
    Compute cost-recovery factor (CRF) per zone and technology.

    Parameters
    ----------
    net : pypsa.Network
    econ_metrics : pd.DataFrame
        Output of get_metrics.get_economic_metrics(net, generation_techs, storage_techs)
        Must contain columns: "Gross Revenue", "Input Costs", "Capital Costs"
    generation_techs, storage_techs : lists
        Technology carriers considered.

    Returns
    -------
    pd.DataFrame
        Index: zones, Columns: technologies (generation_techs + storage_techs).
        Values: CRF as ratio (e.g., 1.0 == 100%).
    """
    techs = list(generation_techs) + list(storage_techs)
    meta = build_component_metadata(net)

    # Join economic metrics with metadata (only components that have economic metrics)
    econ = econ_metrics.copy()
    econ = econ.loc[econ.index.intersection(meta.index)]
    econ = econ.join(meta[['zone', 'technology']], how='left')

    # Only keep relevant technologies
    econ = econ[econ['technology'].isin(techs)]

    # Aggregate per zone x technology
    agg = econ.groupby(['zone', 'technology'])[['Gross Revenue', 'Input Costs', 'Capital Costs']].sum()

    # Compute CRF safely
    denom = agg['Input Costs'] + agg['Capital Costs']
    crf = agg['Gross Revenue'] / denom.replace({0: np.nan})
    crf = crf.unstack(level='technology').reindex(columns=techs).fillna(np.nan)

    # Ensure zones are rows, techs are columns
    crf = crf.sort_index().copy()
    return crf


def calculate_market_value(n, n_ref, generation_techs, storage_techs):
    """
    Compute technology-specific market value per zone.

    Parameters
    ----------
    n : pypsa.Network
    n_ref : pypsa.Network
    generation_techs : list[str]
    storage_techs : list[str]

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: zones, Values: DataFrame of time series (tech → market value).
    """
    zones = n_ref.buses["zone"].unique()
    market_value_ts = {}

    for zone in zones:
        market_value_ts[zone] = pd.DataFrame(index=n.snapshots)
        zone_buses = n_ref.buses[n_ref.buses["zone"] == zone].index
        price = n.buses_t.marginal_price[zone_buses].mean(axis=1)

        # Generators
        for tech in generation_techs:
            gens = n.generators[(n.generators.carrier == tech) &
                                (n.generators.bus.isin(zone_buses)) &
                                (n.generators.p_nom_opt > 0)].index
            if len(gens) > 0:
                power = n.generators_t.p[gens].sum(axis=1)
                market_value_ts[zone][tech] = (power * price / power).fillna(0)

        # Storage
        for tech in storage_techs:
            stores = n.storage_units[(n.storage_units.carrier == tech) &
                                     (n.storage_units.bus.isin(zone_buses)) &
                                     (n.storage_units.p_nom > 0)].index
            if len(stores) > 0:
                power = n.storage_units_t.p[stores].clip(lower=0).sum(axis=1)
                market_value_ts[zone][tech] = (power * price / power).fillna(0)

    return market_value_ts


def get_generator_surplus(n):
    """
    Calculate generator surplus per bus and snapshot.

    Surplus = (market price - marginal cost) × output.

    Parameters
    ----------
    n : pypsa.Network

    Returns
    -------
    pd.DataFrame
        Index: snapshots, Columns: bus IDs, Values: generator surplus (€).
    """
    surplus_df = pd.DataFrame(index=n.snapshots, columns=n.buses.index, data=0.0)
    gen_by_bus = n.generators.groupby("bus").groups

    for bus, gens in gen_by_bus.items():
        for gen in gens:
            output = n.generators_t.p[gen]
            marginal_cost = (
                n.generators_t.marginal_cost[gen]
                if gen in n.generators_t.marginal_cost
                else n.generators.loc[gen, "marginal_cost"]
            )
            price = n.buses_t.marginal_price[bus]
            surplus_df[bus] += (price - marginal_cost) * output

    return surplus_df
