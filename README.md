# FQI

A research pipeline that combines market-regime detection with deep reinforcement learning for FX trading. It resamples GBP/USD tick data to hourly bars and labels market regimes by clustering return distributions with Wasserstein k-means. It then trains a Deep Q-Network (DQN) to choose between flat, long and short positions and backtests the learned policy.

The pipeline only backtests on historical data. It does not connect to a broker or place trades.

## How it works

[`steps.md`](steps.md) is the design plan for the project. In that plan, Wasserstein k-means regime detection feeds a DQN "meta-controller" that is trained with a composite risk-adjusted reward and walk-forward validation. `pipeline.py` implements a simplified first version of the plan:

1. **Load and resample.** Read a tick CSV (`Timestamp`, `Bid`, `Ask`), take the mid price and resample it to 1-hour OHLC bars.
2. **Features.** Compute the log return, 20-bar realised volatility (rolling standard deviation of returns), the 20-bar SMA and the bar's high–low range (`spread`).
3. **Regime detection.** Slide a 48-bar window over the returns. Cluster the windows into 3 regimes with k-means, using a Sinkhorn approximation of the 2-Wasserstein distance from GeomLoss.
4. **State.** Build `[rv, ret, sma, spread, regime]` for each bar and z-score it with `StandardScaler`.
5. **DQN training.** Train for 20 episodes with these settings:
   - 3 actions: 0 = flat, 1 = long, 2 = short
   - Q-network: an MLP with two hidden layers of 128 units
   - experience replay: 50,000 transitions, batches of 64
   - soft-updated target network (τ = 0.01), γ = 0.99, Adam with learning rate 1e-3
   - ε-greedy exploration, with ε multiplied by 0.97 after each episode

   The reward at each step is the price change in the direction of the position, minus two penalties: one for the bar's range and one for the size of the move.
6. **Per-episode backtest.** After each episode, backtest that episode's actions with backtesting.py (10,000 starting cash, 0.02% commission). Print a composite score along with the profit and ε. The score is 0.7 × profit/1000 + 0.2 × win rate − 0.1 × max drawdown, plus a bonus for profit.
7. **Results.** Backtest the trained network's greedy actions over the full dataset and print the statistics. Then show an equity curve, a confusion matrix of actions against next-bar direction, and the distribution of actions.

### Differences from `steps.md`

| Plan in `steps.md` | Current `pipeline.py` |
|---|---|
| Actions choose among strategies, one for each regime | Actions are flat / long / short |
| Walk-forward train and validation windows | Trains and evaluates on the same data |
| Composite risk-adjusted reward drives learning | Composite score is only printed. Learning uses the per-step reward |
| Suggests an LSTM or Transformer Q-network | MLP |
| RSI among the state features | Not included |

## Tech stack

- Python
- PyTorch
- [GeomLoss](https://www.kernel-operations.io/geomloss/) (Sinkhorn / Wasserstein distance)
- [backtesting.py](https://kernc.github.io/backtesting.py/)
- scikit-learn, pandas, NumPy
- matplotlib, seaborn

## Project structure

```
.
├── pipeline.py       # End-to-end run: data → features → regimes → DQN → backtest and plots
├── steps.md          # Design plan: Wasserstein k-means regimes + DQN meta-controller, with a Q&A on the approach
├── requirements.txt  # Python dependencies
└── .gitignore        # Ignores data/, so input data is not committed
```

## Getting started

You need Python 3 and a tick-data CSV.

```bash
git clone https://github.com/TafaraBean/fqi.git
cd fqi
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Data.** The input path is hard-coded at the bottom of `pipeline.py` as `data/Exness_GBPUSD_2025_06.csv`. Put your file there, or edit `path` to point to another file. The CSV needs `Timestamp`, `Bid` and `Ask` columns. The `data/` folder is git-ignored and is not in the repo.

**pandas version.** The pipeline calls `resample("1H")`, which pandas 3 rejects. Either install pandas 2 (`pip install "pandas<3"`) or change the frequency string to `"1h"`.

Run the pipeline:

```bash
python pipeline.py
```

The script prints one line per training episode, then the final backtest statistics. It then opens the three plots.

## Notes and limitations

- **Results are in-sample only.** The scaler, the regime clusters and the DQN are all fitted on the full dataset, and the final backtest runs on that same data. The backtest figures are therefore optimistic.
- **The regime label is treated as a number.** It is scaled like the other features instead of being one-hot encoded.
- **Cluster centres are not Wasserstein barycentres.** Each centre is updated with an element-wise mean of the raw return windows.
- **Results vary between runs.** No random seeds are set, so the regimes and results change from run to run.
- **The cost penalties are heuristics.** The step reward subtracts 10% of the bar range and 1% of the absolute price move, and it does so at every step, including when the agent is flat.
