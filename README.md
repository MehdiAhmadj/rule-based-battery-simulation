# rule-based-battery-simulation
Rule-based battery storage simulation for wind curtailment utilization using anonymized wind generation and electricity price time series.


Rule-Based Battery Simulation for Wind Curtailment Utilization
==============================================================

This repository provides an anonymized implementation of a simple rule-based
battery operation model for studying the utilization of curtailed wind energy.

The model simulates the operation of a battery energy storage system using a
heuristic dispatch strategy. The battery follows these main rules:

1. Charge from curtailed wind energy when available.
2. Discharge when electricity prices are high.
3. Charge from the grid when electricity prices are low.
4. Respect battery power limits, energy capacity, state-of-charge limits, and
   charge/discharge efficiencies.

The purpose of this repository is to demonstrate the simulation workflow,
battery dispatch logic, curtailment handling, and basic economic evaluation.



Description of the Main Files
=============================

src/rule_based_battery.py
-------------------------

This is the main simulation script.

It performs the following steps:

1. Loads electricity price and wind-speed data.
2. Converts wind speed into an anonymized aggregated wind power profile.
3. Calculates available wind energy after local load consumption.
4. Generates stochastic curtailment events.
5. Simulates rule-based battery operation.
6. Evaluates different battery sizes.
7. Evaluates different electricity price threshold scenarios.
8. Exports the simulation results to Excel files.


src/timeseries_loader.py
------------------------

This file contains helper functions for loading and preparing time-series data.

It reads the electricity price and wind-speed CSV files, detects the relevant
time and value columns, converts the data to datetime format, resamples the data
to hourly resolution, and returns aligned hourly time series.

The main function is:

load_hourly_price_and_windspeed()

This function is used by rule_based_battery.py and normally does not need to be
run directly.


Dataset/electricityprice.csv
------------------------------------

This file contains the example electricity price time series.

Expected columns:

dataset,price[€/MWh]

Example:

dataset,price[€/MWh]
2021-01-01 00:00,50.2
2021-01-01 01:00,48.7
2021-01-01 02:00,45.1


Dataset/windspeed_synthetic.csv
----------------------------------------

This file contains the example synthetic or anonymized wind-speed time series.

Expected columns:

datetime,wind_speed

Example:

datetime,wind_speed
2021-01-01 00:00,6.2
2021-01-01 01:00,6.5
2021-01-01 02:00,7.1


Data Privacy Notice
===================

This repository does not include site-specific wind-park information.

The following information is intentionally not included:

- number of turbines
- turbine types
- turbine locations
- wind-park topology
- electrical network layout
- real turbine allocation
- specific turbine power curves
- measured wind-park production data
- confidential project data

Instead, the model uses an anonymized aggregated wind generation representation.


Installation
============

First, clone or download the repository.

Then open a terminal in the main repository folder:

rule-based-battery-simulation/

Create a virtual environment:

Windows:

python -m venv .venv
.venv\Scripts\activate

Linux/macOS:

python -m venv .venv
source .venv/bin/activate

Install the required Python packages:

pip install -r requirements.txt


Required Packages
=================

The requirements.txt file should contain:

numpy
pandas
openpyxl
xlsxwriter
matplotlib


How to Run the Simulation
=========================

Run the main script from the repository root folder.

Use:

python src/rule_based_battery.py

Important:

Do not run timeseries_loader.py directly. It is a helper module imported by
rule_based_battery.py.


Expected Input Files
====================

Before running the model, make sure the following files exist:

Dataset/electricityprice.csv
Dataset/windspeed_synthetic.csv

The paths in rule_based_battery.py should be:

PRICE_CSV = "Dataset/electricityprice.csv"
WIND_CSV = "Dataset/windspeed_synthetic.csv"


Output Files
============

After running the simulation, result files are saved in:

Results/

or in the output folder defined inside rule_based_battery.py.

The model exports Excel files containing:

- battery size
- battery power
- battery capacity
- charged energy
- discharged energy
- curtailed energy used
- total available curtailed energy
- estimated operational profit
- investment cost
- net profit
- electricity price thresholds
- battery efficiency and SOC parameters


Main Simulation Parameters
==========================

The most important parameters are defined at the beginning of
src/rule_based_battery.py.

Simulation period:

START_TIME = "2021-01-01 00:00"
END_TIME = "2024-12-31 23:59"

Anonymized wind capacity:

TOTAL_WIND_CAPACITY_MW = 50.0

Curtailment fraction:

CURTAIL_FRACTION = 0.4

Fixed local load:

FIXED_LOAD_MW = 1.8

Battery unit size:

POWER_PER_UNIT_MW = 2.0
ENERGY_PER_UNIT_MWH = 4.0

Battery efficiency:

EFF_C = 0.92
EFF_D = 0.92

SOC limits:

SOC_MIN = 0.10
SOC_MAX = 0.90

Battery size range:

MIN_UNITS_TOTAL = 0
MAX_UNITS_TOTAL = 15


Electricity Price Scenarios
===========================

The script evaluates different price-threshold scenarios:

H90_L10:
- discharge above the 90th price percentile
- charge from grid below the 10th price percentile

H70_L30:
- discharge above the 70th price percentile
- charge from grid below the 30th price percentile

H50_L50:
- discharge above the 50th price percentile
- charge from grid below the 50th price percentile


Notes
=====

This model is a rule-based simulation, not an optimization model.

The results should be interpreted as a simplified evaluation of battery dispatch
behavior under different assumptions. The model does not include battery
degradation, grid constraints, forecast uncertainty, balancing markets, or a
detailed turbine-level wind-park model.


Recommended Command Summary
===========================

From the repository root folder, run:

pip install -r requirements.txt

then:

python src/rule_based_battery.py
