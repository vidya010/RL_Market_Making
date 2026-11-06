# Risk-Aware Offline RL for Cryptocurrency Market Making

## Overview

This project studies whether an offline reinforcement-learning agent can learn a market-making strategy from historical cryptocurrency order-book data.

The instrument is **Binance USDⓈ-M BTCUSDT perpetual futures**. Historical Level-2 order-book updates and trade data are replayed in a simulated exchange environment. A **Discrete Conservative Q-Learning (CQL)** agent learns when and how to place bid and ask quotes while accounting for inventory risk, adverse selection and execution costs.

**This is a simulation-only project. It does not connect to a live trading account or place real orders.**

## How it works

1. **Data preparation:** Download historical Level-2 order-book updates and trade records from Tardis.dev, then convert them into the format required by `hftbacktest`.
2. **Historical replay:** Replay market events at their original timestamps. The agent makes decisions once per second in one-hour episodes.
3. **Behaviour-policy dataset:** Run rule-based market-making strategies to create logged state, action and reward transitions for offline training.
4. **CQL training:** Train a 33-action Discrete CQL model using `d3rlpy`. The agent chooses among quote spread, quote skew and order-size combinations, as well as stand-aside, hold and flatten actions.
5. **Evaluation:** Evaluate the selected model on held-out dates and compare it with a fixed-spread market-making strategy and a stand-aside reference.

The reward combines changes in portfolio equity with penalties for inventory exposure, adverse selection and order re-quoting costs.

## Data and experiment setup

- **Market:** Binance USDⓈ-M BTCUSDT perpetual futures
- **Data source:** Tardis.dev
- **Data:** Level-2 order-book snapshots/updates and executed trades
- **Date split:** 12 selected monthly sample days, split chronologically into 8 training, 2 validation and 2 test days
- **Decision interval:** 1 second
- **Episode length:** 1 hour
- **Simulator:** `hftbacktest`
- **Offline RL algorithm:** Discrete Conservative Q-Learning (CQL)

The final reported experiment used nine behaviour policies and approximately 6.21 million transitions. The test covered 48 hourly episodes across two held-out dates.

## Dependencies

The main Python dependencies are:

- Python 3.13
- `hftbacktest==2.4.4`
- `d3rlpy==2.8.1`
- PyTorch
- NumPy (`>=2.0,<2.3`)
- Numba
- Pandas and Polars
- SciPy and scikit-learn
- Matplotlib and Seaborn
- tqdm and requests

The notebook installs the Python packages. For a local environment, the core packages can be installed with:

```bash
python -m pip install "hftbacktest==2.4.4" "d3rlpy==2.8.1" "numpy>=2.0,<2.3" polars pandas scipy scikit-learn matplotlib seaborn tqdm requests
```

A compatible PyTorch build is also required. The GPU is optional; the final reported experiment ran on CPU.

## Running the project

The main entry point is `RL_Market_Making.ipynb`.

1. Open the notebook in **Google Colab** (recommended) or a local Jupyter environment.
2. Run the notebook cells from top to bottom.
3. Allow the notebook to install the required packages and create the project folders.
4. Connect Google Drive when prompted if using Colab. The project stores data, models and results under `MyDrive/RL_Market_Making`.
5. Start with `RUN_MODE = 'pilot'` to check that data preparation, replay, dataset generation, training and evaluation work on a small run.
6. After the pilot succeeds, set `RUN_MODE = 'full'` in the configuration section and run the notebook again for the full experiment.

The notebook creates folders for raw and converted data, the offline dataset, model checkpoints, configuration files, evaluation metrics, figures and replay logs. Long-running stages are designed to reuse files that have already been generated where possible.

### Data access

The notebook downloads the selected market data from Tardis.dev. The configured dates are monthly sample dates. An API key may be needed if you change the configuration to dates that are not available as public sample files. No Binance trading API key is required because the project only replays historical data.

## Reported result

In the final reported test, the CQL strategy returned **−8.23 USDT** after costs, while the selected fixed-spread benchmark returned **+8.08 USDT**. The experiment therefore did not demonstrate that CQL outperformed the simpler benchmark. These results apply to the selected instrument, dates and simulation assumptions; they do not establish performance in live trading.

## Reproducibility note

The archived notebook/helper-code snapshot may not be identical to the exact source used for the final `full_v3_wide33` run. In particular, older archived code contains a seven-policy configuration, while the final run report records nine behaviour policies. To reproduce the final reported metrics, use the matching final notebook, configuration, model checkpoint and run metadata together. Do not assume an older notebook snapshot will reproduce those numbers exactly.
