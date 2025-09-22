#pip install numba bottleneck
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution



# ---------- optional numba accel --------------------------------------
try:
    from numba import njit, prange
except Exception:  # pragma: no cover
    njit = None

def _jit(fn):
    if njit is None:
        return fn
    return njit(cache=False, fastmath=False)(fn)


@njit(cache=False, fastmath=False)
def _next_true_indices_jit(cond):
    n = cond.size
    out = np.empty(n, np.int64)
    next_idx = -1
    for i in range(n - 1, -1, -1):
        if cond[i]:
            next_idx = i
        out[i] = next_idx
    return out

@njit(cache=False, fastmath=False, parallel=True)
def _sum_pairs_pos(A, B, I_pos, J_pos, last):
    total = 0.0
    m = I_pos.size
    for k in prange(m):
        i = I_pos[k]
        j = J_pos[k]
        if j == -1:
            j = last
        total += (A[i] - A[j]) / A[i] + (B[j] - B[i]) / B[i]
    return np.float64(total)

@njit(cache=False, fastmath=False, parallel=True)
def _sum_pairs_neg(A, B, I_neg, J_neg, last):
    total = 0.0
    m = I_neg.size
    for k in prange(m):
        i = I_neg[k]
        j = J_neg[k]
        if j == -1:
            j = last
        total += (A[j] - A[i]) / A[i] + (B[i] - B[j]) / B[i]
    return np.float64(total)

@njit(cache=False, fastmath=False)  # sequential wrapper that builds indices
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
    return np.float64(total)


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
    return np.float64(ret_pos.sum() + ret_neg.sum())

# Choose the fastest available backend
_USE_JIT = njit is not None
def _mean_reversion_sum(Z, A, B, rLimit, r2Limit, limit) -> float:
    if _USE_JIT:
        return np.float64(_mean_reversion_sum_jit(Z, A, B, rLimit, r2Limit, limit))
    else:
        return np.float64(_mean_reversion_sum_numpy(Z, A, B, rLimit, r2Limit, limit))

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

def MeanReversion(rLimit, r2Limit, limit, dfNorm, cols, list_ = False, optimize = True):
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



def objective_safe(x, *args):
    # Ensure 1D 
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64).ravel())
    # Catch floating problems inside your math
    with np.errstate(all='raise'):
        val = objective(x, *args)   # call your original function

    # Coerce to a plain Python float and validate
    val = float(np.asarray(val, dtype=np.float64))
    if not np.isfinite(val):
        # Penalize invalid regions instead of NaN-ing out
        return 1e300
    return val


def optimise_thresholds(r1, r2, l, dfNorm: pd.DataFrame, cols: list[str]):

    Z = np.ascontiguousarray(dfNorm['Zscore'].to_numpy(copy=False), dtype=np.float64)
    A = np.ascontiguousarray(dfNorm[cols[0]].to_numpy(copy=False),   dtype=np.float64)
    B = np.ascontiguousarray(dfNorm[cols[1]].to_numpy(copy=False),   dtype=np.float64)


    bounds = [
    (1.5, r1),   # rLimit
    (1.5, r2),   # r2Limit (must end up > rLimit)
    (-1.5, l)  # limit
    ]

    result = differential_evolution(
        objective,
        bounds=bounds,
        args=(Z, A, B),
        strategy="best1bin",
        popsize=40,
        tol=1e-4,
        maxiter=1000,
        polish=True,
        seed=42,
        disp=False,
        workers=1,
        updating='deferred',

    )

    best_rLimit, best_r2Limit, best_limit = result.x
    best_profit = -result.fun


    return np.float64(best_rLimit), np.float64(best_r2Limit), np.float64(best_limit), np.float64(best_profit)

def get_zLimits(r1, r2 ,l, train: pd.DataFrame):
    cols = train.columns.tolist()
    best_rLimit, best_r2Limit, best_limit, best_profit = optimise_thresholds(r1, r2, l, train, cols)
    return best_rLimit, best_r2Limit, best_limit, best_profit

def getAllzlimits(r1, r2, l, roll_window, trainCandles: pd.DataFrame, highCorr: pd.DataFrame, save = False):
    
    rLimits, r2Limits, limits, best_profits, CoinAs, CoinBs = [], [], [], [], [], []

    for r in highCorr.itertuples(index=True):
        try:
            coinPair = trainCandles[[r.CoinA, r.CoinB]].copy()


            coinPair['Spread'] = coinPair[r.CoinA] - coinPair[r.CoinB]
            spread_mean = coinPair['Spread'].rolling(5, min_periods=5).mean()
            sigma = spread_mean.rolling(roll_window).std()
            coinPair['Zscore'] = (coinPair['Spread'] - spread_mean) / sigma
            coinPair.dropna(inplace=True)

            rLimit, r2Limit, limit, best_profit = get_zLimits(r1, r2 ,l, coinPair)

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
    })
    
    if save:
        zlims.to_csv('zlimits.csv', index=False)
    return zlims


def getSignals(zlims, candles, roll_window, save = False):
    dfs = []
    for _, r in zlims.iterrows():
        if r.coinA not in candles.columns or r.coinB not in candles.columns:
            print(f"⛔ Skipping: {r.coinA}-{r.coinB} not in candles")
            continue
        coinPair = candles[[r.coinA, r.coinB]].copy()
        
        
        coinPair['Spread'] = coinPair[r.coinA] - coinPair[r.coinB]
        coinPair['SpreadMean'] = coinPair['Spread'].rolling(5).mean()
        sigma = coinPair['SpreadMean'].rolling(roll_window).std()
        coinPair['Zscore'] = (coinPair['Spread'] - coinPair['SpreadMean'])/sigma
        coinPair.dropna(inplace=True)

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
    final.dropna(inplace=True)
    
    if save:
        final.to_csv('signals_test.csv')
    


    return final
if __name__ == '__main__':
    corr = pd.read_csv('Output/zlimSymbols.csv')
    candles = pd.read_csv('Output/hist_candles_1H.csv')
    print(candles)
    getAllzlimits(trainCandles=candles, highCorr=corr)
