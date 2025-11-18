import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# ===============================================================
# 1. LOAD TICK DATA
# ===============================================================
def load_exness_ticks_multi(data_folder, base_files):
    dfs = []
    for base in base_files:
        feather = os.path.join(data_folder, base + ".feather")
        parquet = os.path.join(data_folder, base + ".parquet")
        csv     = os.path.join(data_folder, base + ".csv")

        if os.path.exists(feather):
            print("Loading FEATHER:", feather)
            df = pd.read_feather(feather)
        elif os.path.exists(parquet):
            print("Loading PARQUET:", parquet)
            df = pd.read_parquet(parquet)
        else:
            print("Loading CSV:", csv)
            df = pd.read_csv(csv)
            df.to_feather(feather)
            df.to_parquet(parquet, index=False)
            print("Saved optimized formats.")

        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df = df.rename(columns={"Timestamp":"timestamp","Bid":"bid","Ask":"ask"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    df = df.dropna(subset=["timestamp","bid","ask"])
    df = df[(df["bid"]>0) & (df["ask"]>0) & (df["ask"]>=df["bid"])]
    df = df.drop_duplicates("timestamp")
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Loaded ticks: {len(df):,}")
    return df


# ===============================================================
# 2. RESAMPLE TICKS → 1-MIN OHLCV
# ===============================================================
def resample_ticks_to_m1(df):
    df = df.copy()
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df = df.set_index("timestamp")

    ohlc = df["mid"].resample("1min").agg(
        open="first", high="max", low="min", close="last"
    )
    volume = df["mid"].resample("1min").count().rename("volume")

    return pd.concat([ohlc, volume], axis=1).ffill().bfill()


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

    # --- Volatility expansion ---
    vol_s = log_ret.rolling(20).std().fillna(0)
    vol_l = log_ret.rolling(60).std().fillna(0)
    df["vol_expansion"] = vol_s / (vol_l + 1e-12)

    df["high_vol_flag"] = (df["vol_expansion"] > 1.5).astype(int)

    # Distribution stats (rolling window)
    win = 30
    m, s, sk, ku = rolling_stats(log_ret.values, win)
    df["dist_mean"] = m
    df["dist_std"]  = s
    df["dist_skew"] = sk
    df["dist_kurt"] = ku

    # Up/down dominance ratio
    pos = log_ret.clip(lower=0)
    neg = (-log_ret).clip(lower=0)
    df["dom_ratio"] = pos.rolling(win).mean().fillna(0) / (neg.rolling(win).mean().fillna(0) + 1e-12)

    # Bias
    ema10 = close.ewm(span=10).mean()
    ema50 = close.ewm(span=50).mean()
    df["bias"] = ema10 - ema50

    # Time features (UTC)
    h = df.index.hour
    df["hour_sin"] = np.sin(2*np.pi*(h/24))
    df["hour_cos"] = np.cos(2*np.pi*(h/24))

    # Sessions
    df["asia"]   = ((h>=0)  & (h<8)).astype(int)
    df["london"] = ((h>=8)  & (h<16)).astype(int)
    df["ny"]     = ((h>=13) & (h<22)).astype(int)

    # ATR (for reward shaping)
    tr = (df["high"] - df["low"]).abs()
    df["atr"] = tr.rolling(30).mean().bfill()

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
    enriched = add_features(df)   # contains atr, dist stats, sessions, etc.

    feat_cols = [
        "vol_expansion","high_vol_flag",
        "dist_mean","dist_std","dist_skew","dist_kurt",
        "dom_ratio","bias",
        "hour_sin","hour_cos",
        "asia","london","ny",
    ]

    feat = normalize(enriched[feat_cols])

    return feat, enriched     # <---- return 2 things



# ===============================================================
# 5. REWARD SHAPING (ATR-Based 1:2 R:R)
# ===============================================================
def compute_rewards(df, actions):
    close = df["close"].values
    price_change = np.diff(close, prepend=close[0])

    # Conservative position scaling
    pnl = actions * price_change

    atr = df["atr"].values
    atr[atr == 0] = 1e-12

    # Risk:Reward = 1:2
    reward = np.where(
        pnl < -atr,       -atr,      # stop loss at -1R
        np.where(
            pnl > 2*atr,  2*atr,     # take profit at +2R
            pnl
        )
    )
    return reward


# ===============================================================
# 6. TRANSITIONS
# ===============================================================
def build_transitions(feat, actions, rewards):
    s  = feat.iloc[:-1].values
    sp = feat.iloc[1:].values
    a  = actions[:-1]
    r  = rewards[1:]
    return s,a,r,sp


# ===============================================================
# 7. FQI EXPERT (XGBoost)
# ===============================================================
class FQIExpert:
    def __init__(self, rho, iters, params):
        self.rho = rho
        self.iters = iters
        self.params = params
        self.gamma = 0.99
        self.actions = [-3,-2,-1,0,1,2,3]
        self.models = {a:None for a in self.actions}

    def fit(self, s, a, r, sp):
        N = len(s)
        idx_map = {act: np.where(a==act)[0] for act in self.actions}

        for it in range(self.iters):
            print(f"[FQI] iteration {it+1}/{self.iters}")

            if it == 0:
                max_next = np.zeros(N)
            else:
                preds = np.vstack([self.models[x].predict(sp) for x in self.actions])
                max_next = preds.max(axis=0)

            for act in self.actions:
                idx = idx_map[act]
                if len(idx)==0: continue

                targets=[]
                for t in idx:
                    total, disc = 0,1
                    for k in range(self.rho):
                        if t+k>=len(r): break
                        total += disc*r[t+k]
                        disc *= self.gamma
                    if t+self.rho < len(max_next):
                        total += disc * max_next[t+self.rho]
                    targets.append(total)

                model = XGBRegressor(**self.params)
                model.fit(s[idx], np.array(targets))
                self.models[act] = model

    def act(self, state):
        return max(self.actions, key=lambda a: self.models[a].predict(state)[0])


# ===============================================================
# 8. TRAIN MULTIPLE EXPERTS
# ===============================================================
def generate_configs():
    cfg=[]
    for rho,iters in [(5,12),(10,12),(20,12),(40,12),(80,12),(300,15)]:
        params = {
            "tree_method":"hist",
            "device":"cuda",  # auto fall back to CPU if no GPU
            "n_estimators":200,
            "max_depth":6,
            "learning_rate":0.05,
        }
        cfg.append((rho,iters,params))
    return cfg


def train_experts(s,a,r,sp):
    experts=[]
    for rho,iters,params in generate_configs():
        print(f"\n=== Training Expert rho={rho}, iters={iters} ===")
        e = FQIExpert(rho,iters,params)
        e.fit(s,a,r,sp)
        experts.append(e)
    return experts


# ===============================================================
# 9. VALIDATION + OAMP BACKTEST
# ===============================================================
def evaluate_expert(e, val_f, val_close):
    eq=0
    for i in range(1,len(val_f)):
        act = e.act(val_f.iloc[i:i+1])
        eq += act * (val_close.iloc[i] - val_close.iloc[i-1])
    return eq


def select_best(experts, val_f, val_close, k=2):
    scores=[]
    for i,e in enumerate(experts):
        print("Evaluating expert", i)
        scores.append((evaluate_expert(e,val_f,val_close), i))

    scores.sort(reverse=True)
    return [experts[i] for _,i in scores[:k]]


class OAMP:
    def __init__(self,K):
        self.weights=np.ones(K)/K
        self.eta=1.0

    def update(self, losses):
        w=self.weights*np.exp(-self.eta*np.array(losses))
        self.weights=w/w.sum()


def compute_daily_actions(experts, states):
    E,T=len(experts),len(states)
    A=np.zeros((E,T))
    for ei,exp in enumerate(experts):
        preds=np.vstack([exp.models[a].predict(states) for a in exp.actions])
        A[ei]=np.argmax(preds,axis=0)-3
    return A


def backtest_oamp(experts, test_states, test_close):
    oamp=OAMP(len(experts))
    eq=[]
    first=True

    df=test_states.copy()
    df["close"]=test_close

    for _,day in df.groupby(df.index.date):

        if not first:
            oamp.update(-prev_returns)
        first=False

        states=day.drop(columns=["close"]).values
        close = day["close"].values

        A = compute_daily_actions(experts, states)
        pc = np.diff(close, prepend=close[0])

        prev_returns = (A*pc).mean(axis=1)
        eq.extend(np.cumsum((A.T @ oamp.weights) * pc))

    return np.array(eq)


# ===============================================================
# 10. MAIN
# ===============================================================
if __name__ == "__main__":

    ticks = load_exness_ticks_multi("data", ["Exness_GBPUSD_2021"])
    m1 = resample_ticks_to_m1(ticks)

    features, enriched = build_feature_matrix(m1)
    closes = enriched["close"]   # safer

    N=len(features)
    N_train=int(N*0.60)
    N_val=int(N*0.20)

    train_f = features.iloc[:N_train]
    val_f   = features.iloc[N_train:N_train+N_val]
    test_f  = features.iloc[N_train+N_val:]

    train_close = closes.iloc[:N_train]
    val_close   = closes.iloc[N_train:N_train+N_val]
    test_close  = closes.iloc[N_train+N_val:]

    # Random behavior policy
    actions=np.random.choice([-3,-2,-1,0,1,2,3], len(train_f))

    rewards = compute_rewards(enriched.loc[train_f.index], actions)


    s,a,r,sp = build_transitions(train_f,actions,rewards)

    experts = train_experts(s,a,r,sp)
    best = select_best(experts,val_f,val_close,k=2)

    eq = backtest_oamp(best,test_f,test_close)

    plt.figure(figsize=(12,5))
    plt.plot(eq)
    plt.title("OAMP Equity Curve — 1-Minute FX RL Pipeline (ATR-Shaped R:R)")
    plt.grid(True)
    plt.show()
