# -*- coding: utf-8 -*-
"""
-----------------------------------------

This script implements a simple non-optimized battery dispatch model.

The battery operation follows a heuristic rule:

    1. Charge from curtailed wind energy first
    2. Discharge when electricity price is high
    3. Charge from the grid when electricity price is low
    4. Respect battery power, energy capacity, SOC limits, and efficiency losses

Important:
----------
This public version intentionally does NOT include site-specific wind-park details,
such as:

    - number of turbines
    - turbine types
    - turbine locations
    - wind-park topology
    - real turbine allocation
    - site-specific turbine power curves

Instead, wind generation is represented by an anonymized aggregated wind power model.

Author: Mehdi Ahmadi
"""

import numpy as np
import pandas as pd
from pathlib import Path

from timeseries_loader import load_hourly_price_and_windspeed


# ============================================================
# -------------------- USER PARAMETERS -----------------------
# ============================================================

START_TIME = "2021-01-01 00:00"
END_TIME = "2024-12-31 23:59"

RANDOM_SEED = 42

# Curtailment settings
CURTAIL_FRACTION = 0.4

# It does not represent the real wind park.
TOTAL_WIND_CAPACITY_MW = 50.0

# Fixed local load consumed before available wind is considered for curtailment/storage
FIXED_LOAD_MW = 1.8

# Battery unit parameters
POWER_PER_UNIT_MW = 2.0
ENERGY_PER_UNIT_MWH = 4.0

# Battery efficiency and SOC limits
EFF_C = 0.92
EFF_D = 0.92
SOC_MIN = 0.10
SOC_MAX = 0.90

# Battery investment cost assumptions
COST_MAP = {
    "p1": 100.76 * 1000 * 4,
    "p2": 128.788 * 1000 * 4,
    "p4": 184.94 * 1000 * 4,
}

COST_PER_UNIT = COST_MAP["p2"]

# Price-threshold scenarios
threshold_scenarios = [
    ("H90_L10", 90, 10),
    ("H70_L30", 70, 30),
    ("H50_L50", 50, 50),
]

# Battery size range
MIN_UNITS_TOTAL = 0
MAX_UNITS_TOTAL = 15

# Input data
PRICE_CSV = "Dataset/electricityprice.csv"
WIND_CSV = "Dataset/windspeed.csv"

# Output folder
OUTPUT_DIR = Path("Results/Rule_Based_Battery")


# ============================================================
# ---------------- MODEL --------------------
# ============================================================

def normalized_wind_power_curve(wind_speed):
    """
    Generic normalized wind power curve.

    This function does not represent a specific turbine model.
    It converts wind speed to normalized power output between 0 and 1.

    Parameters
    ----------
    wind_speed : float
        Wind speed in m/s.

    Returns
    -------
    float
        Normalized wind power output between 0 and 1.
    """

    cut_in = 3.0
    rated = 12.0
    cut_out = 25.0

    if wind_speed < cut_in or wind_speed >= cut_out:
        return 0.0
    elif wind_speed < rated:
        return ((wind_speed - cut_in) / (rated - cut_in)) ** 3
    else:
        return 1.0


def build_aggregated_wind_power_MW(wind_speeds, total_capacity_mw):
    """
    Build an anonymized aggregated wind generation profile.

    No turbine number, turbine type, or wind-park topology is used.

    Parameters
    ----------
    wind_speeds : array-like
        Hourly wind speed values in m/s.

    total_capacity_mw : float
        Aggregated anonymized installed wind capacity in MW.

    Returns
    -------
    np.ndarray
        Aggregated wind generation profile in MW.
    """

    return np.array([
        total_capacity_mw * normalized_wind_power_curve(ws)
        for ws in wind_speeds
    ])


# ============================================================
# ---------------- CURTAILMENT MODEL -------------------------
# ============================================================

def curtail_prob_from_wind(power_frac):
    """
    Map normalized wind power output to curtailment probability.

    Parameters
    ----------
    power_frac : float
        Wind power fraction between 0 and 1.

    Returns
    -------
    float
        Curtailment probability between 0 and 1.
    """

    if power_frac <= 0.4:
        return 0.0428
    elif power_frac <= 0.6:
        return 0.1539
    elif power_frac <= 0.8:
        return 0.3627
    else:
        return 0.6983


def generate_stochastic_curtailment(total_wind_mw, total_capacity_mw, curtail_fraction, random_seed):
    """
    Generate stochastic curtailment events based on aggregated wind output.

    Parameters
    ----------
    total_wind_mw : np.ndarray
        Aggregated wind generation profile in MW.

    total_capacity_mw : float
        Aggregated anonymized installed wind capacity in MW.

    curtail_fraction : float
        Fraction of wind curtailed when a curtailment event occurs.

    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        flags, curtailed_frac, prob_by_hour
    """

    rng = np.random.default_rng(random_seed)

    prob_by_hour = []
    for t in range(len(total_wind_mw)):
        power_frac = min(total_wind_mw[t] / total_capacity_mw, 1.0)
        prob_by_hour.append(curtail_prob_from_wind(power_frac))

    flags = np.array([
        rng.random() < prob_by_hour[t]
        for t in range(len(total_wind_mw))
    ])

    curtailed_frac = np.where(flags, curtail_fraction, 0.0)

    return flags, curtailed_frac, np.array(prob_by_hour)


# ============================================================
# ---------------- BATTERY SIMULATION ------------------------
# ============================================================

def simulate_battery(
    prices,
    wind_MW,
    curtailed_frac,
    units_total,
    price_high_percentile,
    price_low_percentile,
):
    """
    Simulate simple rule-based battery dispatch.

    Parameters
    ----------
    prices : np.ndarray
        Electricity prices in EUR/MWh.

    wind_MW : np.ndarray
        Available wind power in MW.

    curtailed_frac : np.ndarray
        Curtailment fraction for each hour.

    units_total : int
        Number of battery units.

    price_high_percentile : float
        Percentile used to define high-price threshold.

    price_low_percentile : float
        Percentile used to define low-price threshold.

    Returns
    -------
    dict
        Simulation results.
    """

    delta_t_h = 1.0

    power_mw = POWER_PER_UNIT_MW * units_total
    capacity_mwh = ENERGY_PER_UNIT_MWH * units_total

    capacity_kwh = capacity_mwh * 1000
    power_kwh_per_h = power_mw * 1000 * delta_t_h

    # No battery installed
    if units_total == 0:
        return {
            "soc_series_mwh": [0.0] * len(prices),
            "charged_mwh": 0.0,
            "discharged_mwh": 0.0,
            "curtailed_used_mwh": 0.0,
            "profit_eur": 0.0,
            "actions": ["no_battery"] * len(prices),
            "high_price": np.percentile(prices, price_high_percentile),
            "low_price": np.percentile(prices, price_low_percentile),
        }

    soc = 0.5 * capacity_kwh
    soc_min = SOC_MIN * capacity_kwh
    soc_max = SOC_MAX * capacity_kwh

    charged_kwh = 0.0
    discharged_kwh = 0.0
    curtailed_used_kwh = 0.0
    value_saved_eur = 0.0

    soc_series = []
    action_series = []

    high_price = np.percentile(prices, price_high_percentile)
    low_price = np.percentile(prices, price_low_percentile)

    for t in range(len(prices)):
        price = prices[t]
        wind = wind_MW[t]

        curtailed_power_mw = wind * curtailed_frac[t]
        available_curtail_kwh = curtailed_power_mw * 1000 * delta_t_h

        action = "idle"

        # Priority 1: charge from curtailed wind
        if available_curtail_kwh > 0 and soc < soc_max:
            charge_kwh = min(
                power_kwh_per_h,
                available_curtail_kwh,
                (soc_max - soc) / EFF_C,
            )

            soc += charge_kwh * EFF_C
            charged_kwh += charge_kwh
            curtailed_used_kwh += charge_kwh

            action = "charge_curtail"

        # Priority 2: discharge at high price
        elif soc > soc_min and price >= high_price:
            discharge_kwh = min(
                power_kwh_per_h,
                soc - soc_min,
            )

            energy_out_kwh = discharge_kwh * EFF_D
            soc -= discharge_kwh

            discharged_kwh += energy_out_kwh
            value_saved_eur += energy_out_kwh * (price / 1000.0)

            action = "discharge_price"

        # Priority 3: charge from grid at low price
        elif soc < soc_max and price <= low_price:
            charge_kwh = min(
                power_kwh_per_h,
                (soc_max - soc) / EFF_C,
            )

            soc += charge_kwh * EFF_C
            charged_kwh += charge_kwh

            value_saved_eur -= charge_kwh * (price / 1000.0)

            action = "charge_grid"

        soc_series.append(soc / 1000.0)
        action_series.append(action)

    return {
        "soc_series_mwh": soc_series,
        "charged_mwh": charged_kwh / 1000.0,
        "discharged_mwh": discharged_kwh / 1000.0,
        "curtailed_used_mwh": curtailed_used_kwh / 1000.0,
        "profit_eur": value_saved_eur,
        "actions": action_series,
        "high_price": high_price,
        "low_price": low_price,
    }


# ============================================================
# ---------------- EXPORT FUNCTIONS --------------------------
# ============================================================

def save_summary_to_excel(df_summary, file_path):
    """
    Save summary results to Excel.

    Parameters
    ----------
    df_summary : pd.DataFrame
        Summary table.

    file_path : pathlib.Path
        Output Excel file path.
    """

    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)


def save_timeseries_to_excel(
    prices,
    total_wind,
    net_wind,
    curtailed_frac,
    flags,
    prob_by_hour,
    result,
    start_time,
    file_path,
):
    """
    Optional detailed time-series export for one simulation case.

    This output still uses anonymized aggregated wind data.
    """

    T = len(prices)
    hours = np.arange(T)

    time_values = [
        start_time + pd.Timedelta(hours=int(h))
        for h in hours
    ]

    df_timeseries = pd.DataFrame({
        "datetime": time_values,
        "price_EUR_per_MWh": prices,
        "aggregated_wind_MW": total_wind,
        "available_wind_after_load_MW": net_wind,
        "curtailment_probability": prob_by_hour,
        "curtailment_flag": flags.astype(int),
        "curtailment_fraction": curtailed_frac,
        "battery_SOC_MWh": result["soc_series_mwh"],
        "battery_action": result["actions"],
    })

    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        df_timeseries.to_excel(writer, sheet_name="Timeseries", index=False)


# ============================================================
# ---------------- MAIN SCRIPT -------------------------------
# ============================================================

def main():
    print("\n=== rule-based battery simulation ===")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load electricity price and wind-speed data
    # --------------------------------------------------------

    print("\n=== Loading hourly electricity price and wind-speed data ===")

    data = load_hourly_price_and_windspeed(
        price_csv=PRICE_CSV,
        wind_csv=WIND_CSV,
        mode="range",
        start=START_TIME,
        end=END_TIME,
        price_time_col="dataset",
        price_col_candidates=("price[€/MWh]", "price"),
        wind_time_col="datetime",
        wind_col_candidates=("wind_speed", "windspeed"),
    )

    prices = np.array(data["PRICES_EUR_PER_MWH"])
    wind_speeds = np.array([
        0.92 * float(x)
        for x in data["WIND_SPEEDS"]
    ])

    T_HOURS = data["T_HOURS"]

    print(
        f"Loaded {T_HOURS} hourly points "
        f"from {data['meta']['start']} to {data['meta']['end']}"
    )

    # --------------------------------------------------------
    # Build anonymized aggregated wind generation
    # --------------------------------------------------------

    total_wind = build_aggregated_wind_power_MW(
        wind_speeds=wind_speeds,
        total_capacity_mw=TOTAL_WIND_CAPACITY_MW,
    )

    # Local load is consumed first
    net_wind = np.maximum(total_wind - FIXED_LOAD_MW, 0.0)

    # --------------------------------------------------------
    # Generate stochastic curtailment
    # --------------------------------------------------------

    print("\n=== Generating stochastic curtailment ===")

    flags, curtailed_frac, prob_by_hour = generate_stochastic_curtailment(
        total_wind_mw=net_wind,
        total_capacity_mw=TOTAL_WIND_CAPACITY_MW,
        curtail_fraction=CURTAIL_FRACTION,
        random_seed=RANDOM_SEED,
    )

    curtailed_MW = net_wind * curtailed_frac
    curtailed_MWh = curtailed_MW * 1.0
    total_curtailed_energy_MWh = curtailed_MWh.sum()

    print("\n=== Curtailment summary ===")
    print(f"Total curtailed energy: {total_curtailed_energy_MWh:.2f} MWh")

    if net_wind.sum() > 0:
        share_curtailed = 100 * total_curtailed_energy_MWh / net_wind.sum()
    else:
        share_curtailed = 0.0

    print(f"Share of curtailed energy vs available wind: {share_curtailed:.2f}%")

    # Diagnostic print for first 24 hours
    print("\n=== First 24 hours diagnostic ===")
    for t in range(min(T_HOURS, 24)):
        print(
            f"Hour {t:02d}: "
            f"wind={total_wind[t]:.2f} MW, "
            f"net_wind={net_wind[t]:.2f} MW, "
            f"P(curtail)={prob_by_hour[t]:.3f}, "
            f"flag={flags[t]}, "
            f"curtail_frac={curtailed_frac[t]:.2f}"
        )

    # --------------------------------------------------------
    # Prepare time information
    # --------------------------------------------------------

    if "meta" in data and "start" in data["meta"]:
        start_time = pd.Timestamp(data["meta"]["start"])
    else:
        start_time = pd.Timestamp(START_TIME)

    date_str = start_time.strftime("%Y-%m-%d")

    # --------------------------------------------------------
    # Run scenario analysis
    # --------------------------------------------------------

    print("\n=== Running battery-size and threshold-scenario analysis ===")

    all_results = []

    for units_total in range(MIN_UNITS_TOTAL, MAX_UNITS_TOTAL + 1):
        print(f"\n--- Battery units: {units_total} ---")

        power_mw = POWER_PER_UNIT_MW * units_total
        capacity_mwh = ENERGY_PER_UNIT_MWH * units_total
        total_cost = units_total * COST_PER_UNIT

        summary_rows = []

        for scenario_name, high_pct, low_pct in threshold_scenarios:
            print(
                f"Scenario {scenario_name}: "
                f"high percentile={high_pct}, low percentile={low_pct}"
            )

            result = simulate_battery(
                prices=prices,
                wind_MW=net_wind,
                curtailed_frac=curtailed_frac,
                units_total=units_total,
                price_high_percentile=high_pct,
                price_low_percentile=low_pct,
            )

            row = {
                "Scenario": scenario_name,
                "High_pct": high_pct,
                "Low_pct": low_pct,

                "UNITS_TOTAL": units_total,
                "Battery_power_MW": power_mw,
                "Battery_capacity_MWh": capacity_mwh,

                "Charged_MWh": result["charged_mwh"],
                "Discharged_MWh": result["discharged_mwh"],
                "Used_curtailed_energy_MWh": result["curtailed_used_mwh"],
                "Total_available_curtailment_MWh": total_curtailed_energy_MWh,

                "Estimated_Profit_EUR": result["profit_eur"],
                "Total_Investment_EUR": total_cost,
                "Net_Profit_EUR": result["profit_eur"] - total_cost,

                "High_price_EUR_per_MWh": result["high_price"],
                "Low_price_EUR_per_MWh": result["low_price"],

                "Power_per_unit_MW": POWER_PER_UNIT_MW,
                "Energy_per_unit_MWh": ENERGY_PER_UNIT_MWH,
                "Unit_cost_EUR": COST_PER_UNIT,

                "Total_wind_capacity_MW_anonymized": TOTAL_WIND_CAPACITY_MW,
                "Fixed_load_MW": FIXED_LOAD_MW,
                "Curtail_fraction": CURTAIL_FRACTION,
                "EFF_C": EFF_C,
                "EFF_D": EFF_D,
                "SOC_MIN": SOC_MIN,
                "SOC_MAX": SOC_MAX,
            }

            summary_rows.append(row)
            all_results.append(row)

        df_summary = pd.DataFrame(summary_rows)

        filename = (
            f"{date_str}_UNITS{units_total}_"
            f"E{ENERGY_PER_UNIT_MWH}_P{POWER_PER_UNIT_MW}.xlsx"
        )

        file_path = OUTPUT_DIR / filename
        save_summary_to_excel(df_summary, file_path)

        print(f"Saved summary: {file_path}")

    # --------------------------------------------------------
    # Save combined summary for all battery sizes
    # --------------------------------------------------------

    df_all = pd.DataFrame(all_results)

    combined_file = OUTPUT_DIR / f"{date_str}_ALL_RESULTS.xlsx"
    save_summary_to_excel(df_all, combined_file)

    print(f"\nSaved combined summary: {combined_file}")

    # --------------------------------------------------------
    # Optional: save one detailed time-series
    # --------------------------------------------------------

    example_units = MAX_UNITS_TOTAL
    example_scenario = threshold_scenarios[1]  # H70_L30

    example_result = simulate_battery(
        prices=prices,
        wind_MW=net_wind,
        curtailed_frac=curtailed_frac,
        units_total=example_units,
        price_high_percentile=example_scenario[1],
        price_low_percentile=example_scenario[2],
    )

    timeseries_file = OUTPUT_DIR / (
        f"{date_str}_TIMESERIES_example_"
        f"UNITS{example_units}_{example_scenario[0]}.xlsx"
    )

    save_timeseries_to_excel(
        prices=prices,
        total_wind=total_wind,
        net_wind=net_wind,
        curtailed_frac=curtailed_frac,
        flags=flags,
        prob_by_hour=prob_by_hour,
        result=example_result,
        start_time=start_time,
        file_path=timeseries_file,
    )

    print(f"Saved example time series: {timeseries_file}")

    print("\n✅ Simulation completed successfully.")


if __name__ == "__main__":
    main()