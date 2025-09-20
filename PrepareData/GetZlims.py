#pip install numba bottleneck
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from PrepareData.CoinTegrate import cointegrate


# ---------- optional numba accel --------------------------------------
try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None

def _jit(fn):
    if njit is None:
        return fn
    return njit(cache=True, fastmath=True)(fn)

# ---------- ultra-fast primitives -------------------------------------

# replace _jit with explicit options for the hot kernels
from numba import njit, prange
# --- replace your JIT section with this ---
from numba import njit, prange

@njit(cache=True, fastmath=True)
def _next_true_indices_jit(cond):
    n = cond.size
    out = np.empty(n, np.int64)
    next_idx = -1
    for i in range(n - 1, -1, -1):
        if cond[i]:
            next_idx = i
        out[i] = next_idx
    return out

@njit(cache=True, fastmath=True, parallel=True)
def _sum_pairs_pos(A, B, I_pos, J_pos, last):
    total = 0.0
    m = I_pos.size
    for k in prange(m):
        i = I_pos[k]
        j = J_pos[k]
        if j == -1:
            j = last
        total += (A[i] - A[j]) / A[i] + (B[j] - B[i]) / B[i]
    return total

@njit(cache=True, fastmath=True, parallel=True)
def _sum_pairs_neg(A, B, I_neg, J_neg, last):
    total = 0.0
    m = I_neg.size
    for k in prange(m):
        i = I_neg[k]
        j = J_neg[k]
        if j == -1:
            j = last
        total += (A[j] - A[i]) / A[i] + (B[i] - B[j]) / B[i]
    return total

@njit(cache=True, fastmath=True)  # sequential wrapper that builds indices
def _mean_reversion_sum_jit(Z, A, B, rLimit, r2Limit, limit):
    n = Z.size
    last = n - 1

    # build masks
    entry_pos = np.empty(n, np.bool_)
    entry_neg = np.empty(n, np.bool_)
    exit_pos_any = np.empty(n, np.bool_)
    exit_neg_any = np.empty(n, np.bool_)
    for i in range(n):
        z = Z[i]
        entry_pos[i]    = z > rLimit
        entry_neg[i]    = z < -rLimit
        exit_pos_any[i] = (z <= limit) or (z >= rLimit)
        exit_neg_any[i] = (z >= limit) or (z <= -r2Limit)

    # earliest exit mapping (sequential suffix scans)
    next_pos = _next_true_indices_jit(exit_pos_any)
    next_neg = _next_true_indices_jit(exit_neg_any)

    # entry indices
    I_pos = np.flatnonzero(entry_pos)
    I_neg = np.flatnonzero(entry_neg)

    # exit indices for those entries (leave -1 if none)
    J_pos = next_pos[I_pos]
    J_neg = next_neg[I_neg]

    # parallel reductions over independent trades
    total = 0.0
    total += _sum_pairs_pos(A, B, I_pos, J_pos, last)
    total += _sum_pairs_neg(A, B, I_neg, J_neg, last)
    return total


# ---------- NumPy fallbacks (used when Numba unavailable) --------------

def _next_true_indices_numpy(cond: np.ndarray) -> np.ndarray:
    n = cond.size
    idx = np.arange(n, dtype=np.int64)
    big = n + n  # sentinel > any valid index
    tmp = np.where(cond, idx, big)
    sufmin = np.minimum.accumulate(tmp[::-1])[::-1]
    out = sufmin.copy()
    out[out >= n] = -1
    return out

def _mean_reversion_sum_numpy(Z: np.ndarray, A: np.ndarray, B: np.ndarray,
                              rLimit: float, r2Limit: float, limit: float) -> float:
    n = Z.size
    entry_pos = Z > rLimit
    entry_neg = Z < -rLimit
    exit_pos_any = (Z <= limit) | (Z >= rLimit)
    exit_neg_any = (Z >= limit) | (Z <= -r2Limit)
    next_pos = _next_true_indices_numpy(exit_pos_any)
    next_neg = _next_true_indices_numpy(exit_neg_any)
    I_pos = np.flatnonzero(entry_pos)
    I_neg = np.flatnonzero(entry_neg)
    J_pos = np.where(next_pos[I_pos] == -1, n - 1, next_pos[I_pos])
    J_neg = np.where(next_neg[I_neg] == -1, n - 1, next_neg[I_neg])
    ret_pos = (A[I_pos] - A[J_pos]) / A[I_pos] + (B[J_pos] - B[I_pos]) / B[I_pos]
    ret_neg = (A[J_neg] - A[I_neg]) / A[I_neg] + (B[I_neg] - B[J_neg]) / B[I_neg]
    return float(ret_pos.sum() + ret_neg.sum())

# Choose the fastest available backend
_USE_JIT = njit is not None
def _mean_reversion_sum(Z, A, B, rLimit, r2Limit, limit) -> float:
    if _USE_JIT:
        return float(_mean_reversion_sum_jit(Z, A, B, rLimit, r2Limit, limit))
    else:
        return _mean_reversion_sum_numpy(Z, A, B, rLimit, r2Limit, limit)

# ---------- full-details lists variant (kept NumPy) --------------------

def _mean_reversion_lists(idx_vals, Z, A, B, rLimit, r2Limit, limit):
    n = Z.size
    entry_pos = Z > rLimit
    entry_neg = Z < -rLimit
    exit_pos_any = (Z <= limit) | (Z >= rLimit)
    exit_neg_any = (Z >= limit) | (Z <= -r2Limit)
    next_pos = _next_true_indices_numpy(exit_pos_any) if _USE_JIT else _next_true_indices_numpy(exit_pos_any)
    next_neg = _next_true_indices_numpy(exit_neg_any) if _USE_JIT else _next_true_indices_numpy(exit_neg_any)
    I_pos = np.flatnonzero(entry_pos)
    I_neg = np.flatnonzero(entry_neg)
    J_pos = np.where(next_pos[I_pos] == -1, n - 1, next_pos[I_pos])
    J_neg = np.where(next_neg[I_neg] == -1, n - 1, next_neg[I_neg])
    ret_pos = (A[I_pos] - A[J_pos]) / A[I_pos] + (B[J_pos] - B[I_pos]) / B[I_pos]
    ret_neg = (A[J_neg] - A[I_neg]) / A[I_neg] + (B[I_neg] - B[J_neg]) / B[I_neg]

    entr_idx = np.concatenate([I_pos, I_neg])
    rets     = np.concatenate([ret_pos, ret_neg])

    coinA_dir = np.concatenate([np.full(I_pos.size, 'Short', dtype=object),
                                np.full(I_neg.size, 'Long',  dtype=object)])
    coinB_dir = np.concatenate([np.full(I_pos.size, 'Long',  dtype=object),
                                np.full(I_neg.size, 'Short', dtype=object)])
    times = idx_vals[entr_idx]

    order = np.argsort(entr_idx, kind='mergesort')
    return (rets[order].tolist(),
            coinA_dir[order].tolist(),
            coinB_dir[order].tolist(),
            times[order].tolist())

# ---------- public API (drop-in) --------------------------------------

def MeanReversion(rLimit, r2Limit, limit, dfNorm, cols, list_ = True, optimize = True):
    """
    Vectorized, fast implementation.
    cols: [coinA_col, coinB_col]
    """
    # Pull arrays once (no copies)
    Z = dfNorm['Zscore'].to_numpy(copy=False)
    A = dfNorm[cols[0]].to_numpy(copy=False)
    B = dfNorm[cols[1]].to_numpy(copy=False)

    if list_:
        idx_vals = dfNorm.index.to_numpy()
        return _mean_reversion_lists(idx_vals, Z, A, B, rLimit, r2Limit, limit)

    # default path used by the optimiser
    return _mean_reversion_sum(Z, A, B, rLimit, r2Limit, limit)


def objective(x, Z, A, B):
    rLimit, r2Limit, limit = x
    # keep your ordering constraint
    if r2Limit <= rLimit:
        return 1e9
    return -_mean_reversion_sum(Z, A, B, rLimit, r2Limit, limit)

def optimise_thresholds(dfNorm: pd.DataFrame, cols: list[str]):
    # Extract arrays ONCE (contiguous, float32 to reduce bandwidth)
    Z = dfNorm['Zscore'].to_numpy(copy=False)
    A = dfNorm[cols[0]].to_numpy(copy=False)
    B = dfNorm[cols[1]].to_numpy(copy=False)
    """Z = np.ascontiguousarray(dfNorm['Zscore'].to_numpy(copy=False), dtype=np.float32)
    A = np.ascontiguousarray(dfNorm[cols[0]].to_numpy(copy=False),   dtype=np.float32)
    B = np.ascontiguousarray(dfNorm[cols[1]].to_numpy(copy=False),   dtype=np.float32)"""

    """# Tighten bounds to data to avoid useless search space
    zmax = float(np.nanmax(Z))
    zmin = float(np.nanmin(Z))
    r_hi = max(1.5, min(10.0, max(abs(zmax), abs(zmin))))
    bounds = [
        (1.5, r_hi),   # rLimit
        (1.5, max(20.0, r_hi + 1.0)),   # r2Limit (> rLimit)
        (-1.5, 1.5)    # limit
    ]"""
    
    bounds = [
    (1.5, 10),   # rLimit
    (1.5, 20),   # r2Limit (must end up > rLimit)
    (-1.5, 1.5)  # limit
    ]

    result = differential_evolution(
        objective,
        bounds=bounds,
        args=(Z, A, B),
        strategy="best1bin",
        popsize=20,         # slightly leaner for speed; bump if you want
        tol=1e-4,
        maxiter=300,        # trimmed a bit — JIT makes each eval cheap anyway
        polish=False,
        seed=42,
        disp=False,
        workers=1,         # safe: objective is pure-NumPy/Numba
        updating='deferred',

    )

    best_rLimit, best_r2Limit, best_limit = result.x
    best_profit = -result.fun

    print(f"✅  Best rLimit  : {best_rLimit:.4f}")
    print(f"✅  Best r2Limit : {best_r2Limit:.4f}")
    print(f"💰  Max Profit  : {best_profit:,.2f}")

    return float(best_rLimit), float(best_r2Limit), float(best_limit), float(best_profit)

def get_zLimits(train: pd.DataFrame, trainStart=None, trainEnd=200, loop_limit=10):
    cols = train.columns.tolist()  # assumes first two are coinA, coinB
    best_rLimit, best_r2Limit, best_limit, best_profit = optimise_thresholds(train, cols)
    return best_rLimit, best_r2Limit, best_limit, best_profit

def getAllzlimits(trainCandles: pd.DataFrame, highCorr: pd.DataFrame):
    
    rLimits, r2Limits, limits, best_profits, CoinAs, CoinBs = [], [], [], [], [], []

    # iterate; DE runs in parallel inside each iteration
    for r in highCorr.itertuples(index=False):
        try:
            coinPair = trainCandles[[r.CoinA, r.CoinB]].copy()

            # Rolling stats in C (accelerated by Bottleneck if installed)
            coinPair['Spread'] = coinPair[r.CoinA] - coinPair[r.CoinB]
            spread_mean = coinPair['Spread'].rolling(5, min_periods=5).mean()
            sigma = spread_mean.rolling(30, min_periods=30).std()
            coinPair['Zscore'] = (coinPair['Spread'] - spread_mean) / sigma
            coinPair.dropna(inplace=True)

            rLimit, r2Limit, limit, best_profit = get_zLimits(coinPair)

            rLimits.append(rLimit); r2Limits.append(r2Limit); limits.append(limit)
            CoinAs.append(r.CoinA); CoinBs.append(r.CoinB); best_profits.append(best_profit)
        except Exception:
            continue

    zlims = pd.DataFrame({
        'coinA': CoinAs,
        'coinB': CoinBs,
        'rLimit': rLimits,
        'r2Limit': r2Limits,
        'limit': limits,
        'bestProfit': best_profits,
        'qty_A': highCorr['qty_A_units'],
        'units_B': highCorr['qty_B_units_beta'],
        'notional_B_usdt_beta': highCorr['notional_B_usdt_beta']
        
    })
    zlims.to_csv('zlimits.csv', index=False)
    return zlims




def getSignals(zlims, candles):
    dfs = []
    idx = []
    for _, r in zlims.iterrows():
        if r.coinA not in candles.columns or r.coinB not in candles.columns:
            print(f"⛔ Skipping: {r.coinA}-{r.coinB} not in candles")
            continue
        coinPair = candles[[r.coinA, r.coinB]].copy()
        
        
        coinPair['Spread'] = coinPair[r.coinA] - coinPair[r.coinB]
        coinPair['SpreadMean'] = coinPair['Spread'].rolling(5).mean()
        sigma = coinPair['SpreadMean'].rolling(30).std()
        coinPair['Zscore'] = (coinPair['Spread'] - coinPair['SpreadMean'])/sigma


        coinPair.dropna(inplace=True)

        
        #zlimits = zlims[zlims['coinA'].str.contains(r.coinA) & zlims['coinB'].str.contains(r.coinB)]
        rLimit = r['rLimit']
        r2Limit = r['r2Limit']
        limit = r['limit']

        

        
        coinPair['Signal'] = coinPair['Zscore'].apply(lambda x:
            1 if x >= rLimit else
            2 if x <= -rLimit else
            3 if x <= -r2Limit else
            3 if x >= r2Limit else
            3 if np.isclose(x, limit, atol = limit * 0.10) else
            0
            )
        
        df_result = pd.DataFrame({
            'action': coinPair.Signal, #[int(nr) for nr in coinPair.Signal]
            'coinA': candles[r.coinA],
            'coinB': candles[r.coinB]
            
        })
        
    
        df_result.columns = pd.MultiIndex.from_product([[f"{coinPair.columns[0]}-{coinPair.columns[1]}"], df_result.columns])
        

        
        dfs.append(df_result)
                

    final = pd.concat(dfs, join='outer', axis=1)
    
    final['time'] = candles.time
    final.set_index('time', inplace=True)
    
    final.to_csv('signals_test.csv')
    


    return final


if __name__ == '__main__':
    
    candles = pd.read_csv('Output/hist_candles_1H.csv', index_col=0, header=[0,1]).xs('close', axis=1, level=1)
    
    candles = pd.DataFrame(candles).dropna()
    BASE_UNITS_A    = None      # e.g., 1.0  (units of CoinA)
    BASE_NOTIONAL_A = 20.0   # e.g., 1000 USDT to allocate to CoinA
    MAX_KEEP        = 200
    MIN_OBS         = 50
    PVAL_THRESH     = 0.05
    
    corrSymbols = cointegrate(candles, pd.read_csv(r'PrepareData/Data/high_corr_symbols.csv').sample(1000))
    
    print(getAllzlimits(trainCandles=candles, highCorr=corrSymbols))
