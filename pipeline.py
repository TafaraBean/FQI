# ============================================================
# FULL REINFORCEMENT LEARNING TRADING PIPELINE
# ============================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from geomloss import SamplesLoss
from backtesting import Backtest, Strategy
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


# ============================================================
# 1. LOAD + RESAMPLE TO 1H
# ============================================================

def load_and_resample(path):
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    df = df.set_index("Timestamp")

    df["mid"] = (df["Bid"] + df["Ask"]) / 2

    h1 = df["mid"].resample("1H").agg(
        open="first", high="max", low="min", close="last"
    ).dropna()

    return h1


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

def compute_features(df):
    df = df.copy()
    df["ret"] = np.log(df["close"] / df["close"].shift(1))
    df["rv"] = df["ret"].rolling(20).std()
    df["sma"] = df["close"].rolling(20).mean()
    df["spread"] = df["high"] - df["low"]
    df = df.dropna()
    return df


# ============================================================
# 3. WASSERSTEIN K-MEANS
# ============================================================

wasserstein = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)

def get_return_windows(df, window=48):
    X = []
    rets = df["ret"].values.astype(np.float32)

    for i in range(window, len(df)):
        X.append(rets[i-window:i])

    return np.array(X, dtype=np.float32)


def wasserstein_kmeans(X, K=3, iters=5):
    centers = [X[np.random.randint(len(X))] for _ in range(K)]

    for _ in range(iters):
        labels = []
        for sample in X:
            dists = [
                wasserstein(
                    torch.tensor(sample).float().unsqueeze(1),
                    torch.tensor(center).float().unsqueeze(1)
                ).item() for center in centers
            ]
            labels.append(np.argmin(dists))

        new_centers = []
        for k in range(K):
            cluster_samples = [x for x, label in zip(X, labels) if label == k]
            new_centers.append(np.mean(cluster_samples, axis=0))

        centers = new_centers

    return np.array(labels)


# ============================================================
# 4. BUILD STATES
# ============================================================

def build_states(df, regimes):
    df = df.iloc[-len(regimes):].copy()
    df.loc[:, "regime"] = regimes

    feats = df[["rv", "ret", "sma", "spread", "regime"]].values
    scaler = StandardScaler()
    feats = scaler.fit_transform(feats)

    return feats, df


# ============================================================
# 5. BACKTEST REWARD ENGINE
# ============================================================

class ActionStrategy(Strategy):
    actions = None

    def init(self):
        pass

    def next(self):
        i = len(self.data.Close) - 1
        action = self.actions[i]

        if action == 1 and not self.position.is_long:
            self.position.close()
            self.buy()

        elif action == 2 and not self.position.is_short:
            self.position.close()
            self.sell()

        elif action == 0:
            self.position.close()


def compute_reward(actions, df):
    df2 = df.copy()
    df2 = df2.iloc[-len(actions):].copy()

    df2 = df2.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close"
    })

    df2["action"] = actions

    ActionStrategy.actions = actions
    bt = Backtest(df2, ActionStrategy, cash=10_000, commission=0.0002)
    stats = bt.run()

    profit = stats["Equity Final [$]"] - 10000
    win_rate = stats["Win Rate [%]"] / 100
    dd = stats["Max. Drawdown [%]"] / 100

    reward = (
          0.7 * (profit / 1000)
        + 0.2 * win_rate
        - 0.1 * dd
    )

    return reward, stats


# ============================================================
# 6. DQN + REPLAY BUFFER
# ============================================================

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class Replay:
    def __init__(self, size=50000):
        self.buffer = deque(maxlen=size)

    def add(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch=64):
        batch = random.sample(self.buffer, batch)
        s, a, r, s2, d = zip(*batch)
        return (np.array(s), a, r, np.array(s2), d)

    def __len__(self):
        return len(self.buffer)


# ============================================================
# 7. TRAIN DQN
# ============================================================

def train_dqn(states, df, episodes=20):
    input_dim = states.shape[1]
    actions = [0, 1, 2]  # 0=flat, 1=long, 2=short

    q = DQN(input_dim, len(actions))
    q_target = DQN(input_dim, len(actions))
    q_target.load_state_dict(q.state_dict())

    opt = optim.Adam(q.parameters(), lr=1e-3)
    gamma = 0.99
    eps = 1.0
    eps_decay = 0.97
    tau = 0.01   # target smoothing

    replay = Replay()

    closes = df["close"].values
    spreads = df["high"].values - df["low"].values  # for cost penalty

    for ep in range(episodes):
        ep_actions = []
        state = states[0]

        for t in range(1, len(states)):
            # ------------------------------
            # ε-greedy exploration
            # ------------------------------
            if np.random.rand() < eps:
                a = np.random.choice(actions)
            else:
                a = torch.argmax(q(torch.tensor(state).float())).item()

            ep_actions.append(a)
            next_state = states[t]

            # ------------------------------
            # STEP REWARD (immediate PnL)
            # ------------------------------
            price_now = closes[t]
            price_prev = closes[t - 1]

            if a == 1:  # long
                step_reward = price_now - price_prev
            elif a == 2:  # short
                step_reward = price_prev - price_now
            else:
                step_reward = 0

            # ------------------------------
            # Costs & risk penalties
            # ------------------------------
            spread_cost = spreads[t] * 0.1         # scaled cost penalty
            vol_penalty = abs(price_now - price_prev) * 0.01
            step_reward -= (spread_cost + vol_penalty)

            # ------------------------------
            # Store transition
            # ------------------------------
            done = (t == len(states) - 1)
            replay.add(state, a, step_reward, next_state, done)
            state = next_state

            # ------------------------------
            # LEARNING STEP
            # ------------------------------
            if len(replay) > 256:
                s, a_batch, r_batch, s2, d = replay.sample()

                s = torch.tensor(s).float()
                s2 = torch.tensor(s2).float()
                r_batch = torch.tensor(r_batch).float()
                a_batch = torch.tensor(a_batch)
                d = torch.tensor(d).float()     # <-- FIXED here

                # Q(s, a)
                q_values = q(s)
                current_q = q_values[range(len(a_batch)), a_batch]

                # target = r + γ * maxQ(s2)
                with torch.no_grad():
                    next_q = q_target(s2).max(dim=1)[0]
                    td_target = r_batch + gamma * next_q * (1.0 - d)

                loss = (current_q - td_target).pow(2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

                # ------------------------------
                # Soft target update (stable)
                # ------------------------------
                for param, target_param in zip(q.parameters(), q_target.parameters()):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

        # =======================================================
        # EPISODE-END REWARD BOOST (profit incentive)
        # =======================================================
        # Align action count
        while len(ep_actions) < len(df):
            ep_actions.append(0)

        epi_reward, stats = compute_reward(ep_actions, df)

        # Strong positive reinforcement for profitable policies
        final_profit = stats["Equity Final [$]"] - 10000
        final_bonus = max(final_profit, 0) * 0.005
        epi_reward += final_bonus

        print(
            f"Episode {ep+1}/{episodes} | "
            f"Reward: {epi_reward:.4f} | Profit: {final_profit:.2f} | "
            f"Epsilon: {eps:.4f}"
        )

        # epsilon decay
        eps *= eps_decay

    return q

# ============================================================
# 8. RESULTS DISPLAY
# ============================================================

def display_results(dqn, states, df):
    print("\n==================== RESULTS ====================\n")

    actions = []
    with torch.no_grad():
        for s in states:
            actions.append(torch.argmax(dqn(torch.tensor(s).float())).item())

    df2 = df.copy()
    df2 = df2.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close"
    })
    df2 = df2.iloc[-len(actions):].copy()
    df2["action"] = actions

    ActionStrategy.actions = actions
    bt = Backtest(df2, ActionStrategy, cash=10_000, commission=0.0002)
    stats = bt.run()

    print(stats)

    plt.figure(figsize=(12,5))
    plt.plot(stats["_equity_curve"]["Equity"])
    plt.title("Equity Curve")
    plt.grid(True)
    plt.show()

    df2["direction"] = np.sign(df2["Close"].diff().shift(-1)).fillna(0)
    df2["pred_direction"] = df2["action"].replace({1:1, 2:-1, 0:0})

    cm = confusion_matrix(df2["direction"], df2["pred_direction"])

    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.title("Action Prediction Confusion Matrix")
    plt.show()

    plt.figure(figsize=(6,4))
    pd.Series(actions).value_counts().sort_index().plot(kind="bar", title="Action Distribution")
    plt.show()

    print("\n========== RESULTS COMPLETE ==========\n")


# ============================================================
# 9. RUN PIPELINE
# ============================================================

path = r"data/Exness_GBPUSD_2025_06.csv"

df = load_and_resample(path)
df = compute_features(df)

X = get_return_windows(df)
regimes = wasserstein_kmeans(X, K=3)

states, aligned_df = build_states(df, regimes)

dqn = train_dqn(states, aligned_df)

display_results(dqn, states, aligned_df)

