import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

# ===============================================================
# 1. LOAD TICK DATA (EXNESS, UTC)
# ===============================================================
def load_exness_ticks_multi(folder, files):
    dfs = []
    for base in files:
        feather = os.path.join(folder, base + ".feather")
        csv     = os.path.join(folder, base + ".csv")

        if os.path.exists(feather):
            df = pd.read_feather(feather)
        else:
            df = pd.read_csv(csv)
            df.to_feather(feather)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df = df.rename(columns={"Timestamp":"timestamp", "Bid":"bid", "Ask":"ask"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    df = df[(df["ask"] >= df["bid"]) & (df["bid"]>0) & (df["ask"]>0)]
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


# ===============================================================
# 2. RESAMPLE TO 1-MIN OHLCV
# ===============================================================
def resample_ticks_to_m1(df):
    df = df.copy()
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df = df.set_index("timestamp")

    ohlc = df["mid"].resample("1min").agg(
        open="first", high="max", low="min", close="last"
    )
    volume = df["mid"].resample("1min").count().rename("volume")
    m1 = pd.concat([ohlc, volume], axis=1).ffill().bfill()
    return m1


# ===============================================================
# 3. FEATURE ENGINEERING
# ===============================================================
def rolling_stats(arr, win):
    s = pd.Series(arr)
    return (
        s.rolling(win).mean().fillna(0),
        s.rolling(win).std().fillna(0),
        s.rolling(win).skew().fillna(0),
        s.rolling(win).kurt().fillna(0),
    )

def add_features(df):
    df = df.copy()
    close = df["close"]

    log_ret = np.log(close / close.shift(1)).fillna(0)

    vol_short = log_ret.rolling(20).std().fillna(0)
    vol_long = log_ret.rolling(60).std().fillna(0)
    df["vol_expansion"] = vol_short / (vol_long + 1e-12)

    df["high_vol_flag"] = (df["vol_expansion"] > 1.5).astype(int)

    mean, std, skew, kurt = rolling_stats(log_ret.values, 30)
    df["dist_mean"] = mean
    df["dist_std"] = std
    df["dist_skew"] = skew
    df["dist_kurt"] = kurt

    pos = log_ret.clip(lower=0)
    neg = (-log_ret).clip(lower=0)
    df["dom_ratio"] = pos.rolling(30).mean() / (neg.rolling(30).mean() + 1e-12)

    ema10 = close.ewm(span=10).mean()
    ema50 = close.ewm(span=50).mean()
    df["bias"] = ema10 - ema50

    h = df.index.hour
    df["hour_sin"] = np.sin(2*np.pi*(h/24))
    df["hour_cos"] = np.cos(2*np.pi*(h/24))

    df["asia"] = ((h>=0)&(h<8)).astype(int)
    df["london"] = ((h>=8)&(h<16)).astype(int)
    df["ny"] = ((h>=13)&(h<22)).astype(int)

    # ATR for reward normalization
    tr = (df["high"] - df["low"]).abs()
    df["atr"] = tr.rolling(30).mean().fillna(method="bfill")

    return df.fillna(0)


# ===============================================================
# 4. NORMALIZATION
# ===============================================================
def normalize(df):
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        x = df[col].astype(float)
        out[col] = 2*(x - x.min())/(x.max() - x.min() + 1e-12) - 1
    return out


def build_feature_matrix(df):
    df = add_features(df)

    feat_cols = [
        "vol_expansion","high_vol_flag",
        "dist_mean","dist_std","dist_skew","dist_kurt",
        "dom_ratio","bias",
        "hour_sin","hour_cos",
        "asia","london","ny"
    ]
    feat = normalize(df[feat_cols])
    return feat


# ===============================================================
# 5. ATR-NORMALIZED REWARD (MODE 2)
# ===============================================================
def compute_rewards(df, actions):
    close = df["close"].values
    pc = np.diff(close, prepend=close[0])

    switch_cost = np.abs(np.diff(actions, prepend=actions[0])) * 0.00001
    raw = actions * pc - switch_cost

    atr = df["atr"].values
    reward = raw / (atr + 1e-12)

    return reward


# ===============================================================
# 6. TRANSITIONS
# ===============================================================
def build_transitions(feat, actions, rewards):
    s = feat.iloc[:-1].values
    sp = feat.iloc[1:].values
    a = actions[:-1]
    r = rewards[1:]
    return s,a,r,sp


# ===============================================================
# 7. FQI EXPERT (CPU XGBoost)
# ===============================================================
class FQIExpert:
    def __init__(self, rho, iters, params):
        self.rho = rho
        self.iters = iters
        self.params = params
        self.actions = [-1,0,1]
        self.gamma = 0.99
        self.models = {a:None for a in self.actions}

    def fit(self, s, a, r, sp):
        N = len(s)
        idx_map = {act: np.where(a==act)[0] for act in self.actions}

        max_next = np.zeros(N)

        for it in range(self.iters):
            print(f"[FQI] iteration {it+1}/{self.iters}")

            if it>0:
                preds = np.vstack([self.models[x].predict(sp) for x in self.actions])
                max_next = preds.max(axis=0)

            for act in self.actions:
                idx = idx_map[act]
                if len(idx)==0: continue

                # Vectorized multi-step target
                targets = []
                for t in idx:
                    end = min(t+self.rho, len(r))
                    steps = np.arange(end - t)
                    disc = self.gamma ** steps
                    total = np.sum(disc * r[t:end])
                    if t+self.rho < len(max_next):
                        total += (self.gamma**self.rho) * max_next[t+self.rho]
                    targets.append(total)

                model = XGBRegressor(**self.params)
                model.fit(s[idx], np.array(targets))
                self.models[act] = model

    def act(self, state):
        return max(self.actions, key=lambda a: self.models[a].predict(state)[0])


# ===============================================================
# 8. MULTIPLE EXPERT CONFIGS
# ===============================================================
def generate_configs():
    cfg = []
    horizons = [5, 15, 30, 60, 120, 300]   # 5 min → 5 hours
    for rho in horizons:
        params = {
            "tree_method":"hist",
            "n_estimators":180,
            "max_depth":6,
            "learning_rate":0.05,
            "nthread":-1
        }
        cfg.append((rho, 12, params))
    return cfg


def train_experts(s,a,r,sp):
    experts=[]
    for rho,iters,params in generate_configs():
        print(f"\n=== Expert rho={rho}, iters={iters} ===")
        e = FQIExpert(rho,iters,params)
        e.fit(s,a,r,sp)
        experts.append(e)
    return experts


# ===============================================================
# 9. VALIDATION
# ===============================================================
def evaluate_expert(e, val_f, val_close):
    eq = 0
    for i in range(1,len(val_f)):
        act = e.act(val_f.iloc[i:i+1])
        eq += act * (val_close.iloc[i] - val_close.iloc[i-1])
    return eq


def select_best(experts, val_f, val_close, k=3):
    scores=[]
    for i,e in enumerate(experts):
        print("Evaluating expert",i)
        scores.append((evaluate_expert(e,val_f,val_close), i))
    scores.sort(reverse=True)
    return [experts[i] for _,i in scores[:k]]


# ===============================================================
# 10. OAMP PORTFOLIO BACKTEST
# ===============================================================
class OAMP:
    def __init__(self,K):
        self.weights = np.ones(K)/K
        self.eta = 1.0

    def update(self, losses):
        w = self.weights * np.exp(-self.eta * np.array(losses))
        self.weights = w / w.sum()


def compute_daily_actions(experts, states):
    E,T = len(experts), len(states)
    A = np.zeros((E,T))
    for ei,exp in enumerate(experts):
        preds = np.vstack([exp.models[a].predict(states) for a in exp.actions])
        A[ei] = np.argmax(preds,axis=0)-1
    return A


def backtest_oamp(experts, test_states, test_close):
    oamp = OAMP(len(experts))
    eq = []
    first=True

    df = test_states.copy()
    df["close"] = test_close

    for _,day in df.groupby(df.index.date):
        if not first:
            oamp.update(-prev_returns)
        first=False

        states = day.drop(columns=["close"]).values
        close = day["close"].values

        A = compute_daily_actions(experts, states)
        pc = np.diff(close, prepend=close[0])

        prev_returns = (A*pc).mean(axis=1)
        eq.extend(np.cumsum((A.T @ oamp.weights) * pc))

    return np.array(eq)


# ===============================================================
# 11. MAIN
# ===============================================================
if __name__ == "__main__":

    # LOAD TICKS
    ticks = load_exness_ticks_multi("data", ["Exness_GBPUSD_2021","Exness_GBPUSD_2022"])
    m1 = resample_ticks_to_m1(ticks)
    m1["atr"] = (m1["high"] - m1["low"]).rolling(30).mean().bfill()


    features = build_feature_matrix(m1)
    closes = m1["close"].loc[features.index]

    N=len(features)
    N_train=int(N*0.60)
    N_val=int(N*0.20)

    train_f = features.iloc[:N_train]
    val_f   = features.iloc[N_train:N_train+N_val]
    test_f  = features.iloc[N_train+N_val:]

    train_close = closes.iloc[:N_train]
    val_close   = closes.iloc[N_train:N_train+N_val]
    test_close  = closes.iloc[N_train+N_val:]

    # ε-greedy behavior policy
    eps = 0.2
    greedy_actions = np.sign(train_f["bias"].values)
    random_actions = np.random.choice([-1,0,1], len(train_f))
    mask = np.random.rand(len(train_f)) < eps
    actions = np.where(mask, random_actions, greedy_actions)

    rewards = compute_rewards(m1.loc[train_f.index], actions)
    s,a,r,sp = build_transitions(train_f,actions,rewards)

    experts = train_experts(s,a,r,sp)
    best = select_best(experts,val_f,val_close,k=3)

    eq = backtest_oamp(best,test_f,test_close)

    plt.figure(figsize=(14,6))
    plt.plot(eq)
    plt.title("Institutional FX RL Model — Multi-Horizon Experts + ATR-Normalized Reward")
    plt.grid(True)
    plt.show()
