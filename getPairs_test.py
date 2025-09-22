import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from PrepareData.getZlims_test import getSignals, getAllzlimits
from PrepareData.getSymbols import getHighCorrSymbols

# Avoid BLAS oversubscription when pandas/NumPy do rollings/corr
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")


@dataclass
class CoinPair:
    CoinA: str
    CoinB: str


def Signals(r1, r2, l, roll_window, traincandles, testcandles, corrCointegrated):
    # unchanged behavior: just forwards to the underlying functions
    zlimits = getAllzlimits(r1, r2, l, roll_window, traincandles, corrCointegrated)
    signals = getSignals(zlimits, testcandles, roll_window)
    return signals


def evaluate_signals(corrCointegrated, signals, max_hold=30) -> pd.DataFrame:
    # Keep original logic; reduce overhead by reusing slices
    df = signals.dropna()
    cols = df.columns.get_level_values(0).unique()

    result = []
    std_ = []
    total_trades = []

    # Pre-extract to avoid recomputing inside loops
    for coin in cols:
        profits = []

        coinA_name, coinB_name = coin.split('-', 1)
        # keep sizing commented as in original
        sizeA = 1
        sizeB = 1

        df_coin = df[coin]  # narrower view (same index as df)
        # Iteration semantics must match original -> keep itertuples + .loc slicing
        for r in df_coin.itertuples():
            if r.action == 1:
                priceA_short = r.coinA * sizeA
                priceB_buy = r.coinB * sizeB
                i = 0
                # slice once
                tail = df_coin.loc[r.Index:]
                for r2 in tail.itertuples():
                    if r2.action == 3:
                        priceA_buy = r2.coinA * sizeA
                        priceB_short = r2.coinB * sizeB
                        profits.append((priceA_short - priceA_buy) / priceA_short - 0.0012)
                        profits.append((priceB_short - priceB_buy) / priceB_buy - 0.0012)
                        break
                    if i >= max_hold:
                        priceA_buy = r2.coinA * sizeA
                        priceB_short = r2.coinB * sizeB
                        profits.append((priceA_short - priceA_buy) / priceA_short - 0.0012)
                        profits.append((priceB_short - priceB_buy) / priceB_buy - 0.0012)
                        break
                    i += 1

            if r.action == 2:
                priceA_buy = r.coinA * sizeA
                priceB_short = r.coinB * sizeB
                i = 0
                tail = df_coin.loc[r.Index:]
                for r2 in tail.itertuples():
                    if r2.action == 3:
                        priceA_short = r2.coinA * sizeA
                        priceB_buy = r2.coinB * sizeB
                        profits.append((priceA_short - priceA_buy) / priceA_buy - 0.0012)
                        profits.append((priceB_short - priceB_buy) / priceB_short - 0.0012)
                        break
                    if i >= max_hold:
                        priceA_short = r2.coinA * sizeA
                        priceB_buy = r2.coinB * sizeB
                        profits.append((priceA_short - priceA_buy) / priceA_buy - 0.0012)
                        profits.append((priceB_short - priceB_buy) / priceB_short - 0.0012)
                        break
                    i += 1

        result.append(sum(profits))
        std_.append(np.std(profits) if profits else 0.0)
        total_trades.append(len(profits))

    df_result = pd.DataFrame(
        {'profit': result, 'std': std_, 'total_trades': total_trades},
        index=cols
    )
    return df_result


def getZcores2(trainCandles: pd.DataFrame, highCorr: list, roll_window: int) -> pd.DataFrame:
    """
    Build z-scores for each CoinA-CoinB column.
    NOTE: keeps original behavior including min_periods=30 for sigma.
    """
    zscores = []
    # Speed: avoid DataFrame copies; compute on arrays then rename to the expected column name
    for r in highCorr:
        a = trainCandles[r.CoinA]
        b = trainCandles[r.CoinB]
        spread = a - b
        spread_mean = spread.rolling(5, min_periods=5).mean()
        sigma = spread_mean.rolling(roll_window, min_periods=30).std()  # unchanged min_periods=30
        zscore = (spread - spread_mean) / sigma
        zscores.append(zscore.rename(f"{r.CoinA}-{r.CoinB}"))

    # Single concat at the end
    return pd.concat(zscores, axis=1)


def getLowestCorrPairs(corr_: pd.DataFrame, k: int = 10, criterion: str = "mean"):
    if corr_.empty:
        return []

    cm = corr_.copy()

    # Collapse duplicates (same as before, but avoid deprecated axis args)
    cm = cm.T.groupby(level=0, sort=False).mean(numeric_only=True).T
    cm = cm.groupby(level=0, sort=False).mean(numeric_only=True)

    cm = cm.reindex(index=cm.columns, columns=cm.columns)
    np.fill_diagonal(cm.values, np.nan)

    valid_cols = cm.columns[~cm.abs().isna().all(axis=0)]
    cm = cm.loc[valid_cols, valid_cols]
    if cm.empty:
        return []

    # lowest mean |corr| seed
    avg_abs = cm.abs().apply(np.nanmean, axis=0).dropna()
    if avg_abs.empty:
        return []
    first = avg_abs.idxmin()

    selected = [first]
    remaining = [c for c in cm.columns if c != first]

    def _score(candidate: str) -> float:
        vals = cm.loc[selected, candidate]
        arr = vals.to_numpy(dtype=float, copy=False).ravel()
        if arr.size == 0 or np.all(np.isnan(arr)):
            return np.inf
        if criterion == "max":
            return float(np.nanmax(arr))
        return float(np.nanmean(arr))

    target = min(k, cm.shape[1])
    while len(selected) < target and remaining:
        next_choice = min(remaining, key=_score)
        selected.append(next_choice)
        if next_choice in remaining:
            remaining.remove(next_choice)

    return selected


def select_pairs(candles: pd.DataFrame, pairs: pd.DataFrame,
                 max_std=0.1, min_profit=0.2, roll_window=30):
    # Keep identical filtering semantics on MultiIndex rows
    filt = (pairs['std'] < max_std) & (pairs['profit'] > min_profit)
    filt_idx = pairs.loc[filt].index.unique().tolist()
    if not filt_idx:
        return pd.DataFrame(columns=['CoinA', 'CoinB'])

    pairs_dataClass = [CoinPair(r[0], r[1]) for r in filt_idx]

    # Build z-scores (unchanged logic)
    zscores = getZcores2(candles, pairs_dataClass, roll_window)
    if zscores.shape[1] == 0:
        return pd.DataFrame(columns=['CoinA', 'CoinB'])

    cormat = zscores.corr(min_periods=1)
    selected_cols = getLowestCorrPairs(cormat, k=10, criterion="mean")
    if not selected_cols:
        return pd.DataFrame(columns=['CoinA', 'CoinB'])

    coinA = [p.split('-', 1)[0] for p in selected_cols]
    coinB = [p.split('-', 1)[1] for p in selected_cols]
    return pd.DataFrame({'CoinA': coinA, 'CoinB': coinB})


def trainSymbols(candles: pd.DataFrame, highCorr_500: pd.DataFrame,
                 max_pairs, max_hold,
                 step, r1=10, r2=15, l=5, roll_window=30,
                 train_rows_start=1000, train_rows_end=2200):

    # Keep exact selection behavior
    if max_pairs != 'all':
        symbols = highCorr_500['CoinA'].sample(max_pairs).unique().tolist()
        print('Total Amount of Unique Symbols:', len(symbols))
    else:
        symbols = highCorr_500['CoinA'].unique().tolist()
        print('Total Amount of Unique Symbols:', len(symbols))

    dfs = []
    last_training_end = None

    # tqdm over symbols (unchanged)
    for _, symbol in zip(tqdm(range(len(symbols))), symbols):
        current_start = train_rows_start

        # Rolling windows loop
        while True:
            next_start = current_start + step
            if next_start >= train_rows_end:
                break

            # Slicing once
            traincandles = candles[current_start:next_start]
            # EXACT original behavior: testcandles uses reset_index + rename
            testcandles = candles[next_start: next_start + step].reset_index().rename(columns={'date': 'time'})

            try:
                highCorr_ = highCorr_500[highCorr_500['CoinA'] == symbol]
                signals = Signals(r1, r2, l, roll_window, traincandles, testcandles, highCorr_)
            except Exception as e:
                current_start = next_start
                print(f"Error occurred: {e}")
                continue

            result = evaluate_signals(highCorr_, signals, max_hold)
            dfs.append(result)

            last_training_end = next_start + step
            current_start = next_start

    if not dfs:
        raise ValueError("No training windows were generated before reaching train_rows_end.")

    df_pairs = pd.concat(dfs)
    

    # EXACT same MultiIndex construction and CSV output
    idx0 = df_pairs.index.str.split('-').str[0]
    idx1 = df_pairs.index.str.split('-').str[1]
    df_pairs.index = pd.MultiIndex.from_arrays([idx0, idx1], names=['CoinA', 'CoinB'])
    df_pairs.to_csv('Output/best_pairs.csv')
    return df_pairs, last_training_end


def testSymbols(candles: pd.DataFrame, pairs: pd.DataFrame,
                r1, r2, l, max_std, min_profit, max_hold=30, roll_window=30,
                testStart=1200, testEnd=1800, step=200):

    result = []
    # EXACT stepping and windowing; only minor micro-opts
    for i in range(testStart, testEnd, step):
        traincandles = candles[i - step:i]
        # original uses i-roll_window:i+step for testcandles (keep)
        testcandles = candles[i - roll_window:i + step].reset_index().rename(columns={'date': 'time'})

        try:
            corr = select_pairs(traincandles, pairs, max_std, min_profit, roll_window)
            signals = Signals(r1, r2, l, roll_window, traincandles, testcandles, corr)
            evalutation = evaluate_signals(corr, signals, max_hold)
            result.append(evalutation)
        except Exception as e:
            print(f"Error occurred: {e}")
            continue

    df_ = pd.concat(result)

    # Keep exact prints/statistics
    print('--- Final Results ---', '\n')
    print('Total Profit:', df_.profit.sum())
    print('Mean Profit per iteration:', df_.groupby(level=0).sum().profit.mean())
    print('Max Standard Deviation of All Pairs:', df_.groupby(level=0).sum()['std'].max())
    print('Total Trades:', df_.total_trades.sum())
    print('Max Hold Time:', max_hold)

    return {
        'Test_Start': testStart,
        'Test_End': testEnd,
        'Total_Profit': float(df_.profit.sum()),
        'Mean_Profit_per_iteration': float(df_.groupby(level=0).sum().profit.mean()),
        'Max_Standard_Deviation_of_All_Pairs': float(df_.groupby(level=0).sum()['std'].max()),
        'Total_Trades': int(df_.total_trades.sum()),
        'Max_Hold_Time': max_hold
    }


if __name__ == "__main__":
    os.makedirs('Test_Output', exist_ok=True)

    # unchanged load and slicing
    candles = pd.read_csv('Output/hist_candles_1H.csv', index_col=0, header=[0, 1]).xs('close', axis=1, level=1)
    # candles = candles.sample(30, axis=1)

    output = {}
    loop = 1

    r1 = 5
    r2 = 15
    l = 5
    max_hold = 10
    roll_window = 30
    max_std = 0.1
    min_profit = -8
    step = 200

    fileName = (f'Test_Output/r1={r1},r2={r2},l={l},max_hold={max_hold},'
                f'roll_window={roll_window},step={step},max_std={max_std},min_profit={min_profit}.json')

    print(len(candles))

    for i in range(1000, 9000, 1000):
        candles_ = candles

        highCorr_500 = getHighCorrSymbols(candles_[i - 1000:i])[:30]

        pairs, last_training_end = trainSymbols(
            candles_, highCorr_500,
            max_pairs='all', max_hold=max_hold, step=step,
            r1=r1, r2=r2, l=l, roll_window=roll_window,
            train_rows_start=i - 1000, train_rows_end=i
        )
        try:
            result = testSymbols(
                candles_, pairs,
                r1=r1, r2=r2, l=l,
                max_std=max_std, min_profit=min_profit,
                max_hold=max_hold, roll_window=roll_window,
                testStart=last_training_end, testEnd=last_training_end + 1000, step=step
            )

            result['train_start'] = i - 1000
            result['train_end'] = i

            output.update({f'Loop_{loop}': result})

            with open(fileName, 'w') as f:
                json.dump(output, f, indent=4)

        except Exception:
            pass

        loop += 1
