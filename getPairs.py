import os
from tqdm import tqdm
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"
os.environ["MKL_THREADING_LAYER"] = "INTEL"   # helps MKL+OpenMP combos
import faulthandler; faulthandler.enable()


import numpy as np
from getSymbols import getSignals, getAllzlimits
import pandas as pd
from PrepareData.CoinTegrate import cointegrate
from dataclasses import dataclass


def getHighCorrSymbols(candles: pd.DataFrame, save: bool = False):

    dfCorr = candles.corr()    
    np.fill_diagonal(dfCorr.values, 0)

    df = pd.DataFrame(dfCorr.where(dfCorr > 0.9).stack().index.tolist(), columns=['CoinA', 'CoinB'])
    
    symbols = df
    if save:
        symbols.to_csv('Output/high_corr_symbols.csv', index=False)
        
    return symbols


def Signals(r2, traincandles, testcandles, corrCointegrated):
        zlimits = getAllzlimits(r2, traincandles, corrCointegrated)
        signals = getSignals(zlimits, testcandles)
        
        return signals
        
        

def evaluate_signals(corrCointegrated, signals, max_hold=30) -> pd.DataFrame:
    df = signals.dropna()
    cols = df.columns.get_level_values(0).unique()

    
    result = []
    std_ = []
    total_trades = []
    
    for coin in cols:
        profits = []
        
        coinA = coin.split('-')[0]
        coinB = coin.split('-')[1]
            
        sizeA = 1#corrCointegrated[(corrCointegrated['CoinA'] == coinA) & (corrCointegrated['CoinB'] == coinB)]['qty_A_units'].iloc[0]
        sizeB = 1#corrCointegrated[(corrCointegrated['CoinA'] == coinA) & (corrCointegrated['CoinB'] == coinB)]['qty_B_units_beta'].iloc[0]
        for r in df[coin].itertuples():
            

            if r.action == 1:

                
                
                priceA_short = r.coinA * sizeA
                priceB_buy = r.coinB * sizeB
                i = 0
                for r2 in df[coin].loc[r.Index:].itertuples():
                    if r2.action == 3:


                        priceA_buy = r2.coinA * sizeA
                        priceB_short = r2.coinB * sizeB
                        
                        profit1 = (priceA_short - priceA_buy) / priceA_short - 0.0012
                        profit2 = (priceB_short - priceB_buy) / priceB_buy - 0.0012
                        profits.append(profit1 + profit2)
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
                for r2 in df[coin].loc[r.Index:].itertuples():
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
        std_.append(np.std(profits) if profits else 0)
        total_trades.append(len(profits))

    df_result = pd.DataFrame({'profit': result, 'std': std_, 'total_trades': total_trades}, index = cols)
    return df_result


def getZcores2(trainCandles: pd.DataFrame, highCorr: list) -> pd.DataFrame:
    """Build z-scores for each CoinA-CoinB column. Robust to NaNs."""
    zscores = []
    for r in highCorr:
        coinPair = trainCandles[[r.CoinA, r.CoinB]].copy()
        spread = coinPair[r.CoinA] - coinPair[r.CoinB]
        spread_mean = spread.rolling(5, min_periods=5).mean()
        sigma = spread_mean.rolling(30, min_periods=30).std()
        zscore = (spread - spread_mean) / sigma
        col_name = f"{r.CoinA}-{r.CoinB}"
        zscores.append(zscore.rename(col_name))

    zscores_df = pd.concat(zscores, axis=1)
    # Drop fully-NA columns (or keep this line commented if you prefer to keep them)
    #zscores_df = zscores_df.dropna(axis=1, how='all')
    
    
    
    return zscores_df

def getLowestCorrPairs(corr_: pd.DataFrame, k: int = 10, criterion: str = "mean"):
    """
    Greedy subset: pick k columns that are minimally correlated with each other.
    - Collapses duplicate labels by averaging (rows/cols) to keep symmetry.
    - Ignores diagonal.
    - criterion: "mean" (avg |corr| to selected) or "max" (max |corr| to selected).
    """
    if corr_.empty:
        return []

    # 1) Work on a copy and collapse duplicates on both axes (keeps it square & symmetric)
    cm = corr_.copy()

    # If you know there are no duplicate labels, you can skip the groupby lines.
    # Collapse duplicate columns
    cm = cm.T.groupby(level=0).mean(numeric_only=True).T
    # Collapse duplicate rows
    cm = cm.groupby(level=0).mean(numeric_only=True)

    cm = cm.reindex(index=cm.columns, columns=cm.columns)

    # 2) Ignore self-correlation
    np.fill_diagonal(cm.values, np.nan)

    # 3) Drop all-NA columns (cannot seed from them)
    valid_cols = cm.columns[~cm.abs().isna().all(axis=0)]
    cm = cm.loc[valid_cols, valid_cols]
    if cm.empty:
        return []

    # 4) Seed with the column with lowest mean |corr| to the universe
    avg_abs = cm.abs().apply(np.nanmean, axis=0).dropna()
    if avg_abs.empty:
        return []
    first = avg_abs.idxmin()

    selected = [first]
    remaining = [c for c in cm.columns if c != first]

    # 5) Scoring helper → always returns a float
    def _score(candidate: str) -> float:
        vals = cm.loc[selected, candidate]
        # Could be DataFrame if duplicate labels sneak in; always reduce to ndarray
        arr = vals.to_numpy(dtype=float) if hasattr(vals, "to_numpy") else np.asarray(vals, dtype=float)
        arr = np.abs(arr).ravel()
        if arr.size == 0 or np.all(np.isnan(arr)):
            return np.inf
        if criterion == "max":
            return float(np.nanmax(arr))
        return float(np.nanmean(arr))

    # 6) Greedy build
    target = min(k, cm.shape[1])
    while len(selected) < target and remaining:
        next_choice = min(remaining, key=_score)
        selected.append(next_choice)
        # Guard against "x not in list"
        if next_choice in remaining:
            remaining.remove(next_choice)

    return selected

def select_pairs(candles: pd.DataFrame, pairs: pd.DataFrame, max_std=0.1, min_profit=0.2):
    from dataclasses import dataclass

    @dataclass
    class CoinPair:
        CoinA: str
        CoinB: str

    # Filter by your thresholds on the MultiIndex rows
    filt_idx = pairs[(pairs['std'] < max_std) & (pairs['profit'] > min_profit)].index.unique().tolist()
    if not filt_idx:
        # Nothing passes the filter; return empty DataFrame with expected columns
        return pd.DataFrame(columns=['CoinA', 'CoinB'])

    pairs_dataClass = [CoinPair(r[0], r[1]) for r in filt_idx]

    # Build z-scores and correlation matrix
    zscores = getZcores2(candles, pairs_dataClass)
    if zscores.shape[1] == 0:
        return pd.DataFrame(columns=['CoinA', 'CoinB'])

    cormat = zscores.corr(min_periods=1)

    # Select up to 10 least-correlated columns (by greedy diversification)
    selected_cols = getLowestCorrPairs(cormat, k=10, criterion="mean")  # or criterion="max"

    if not selected_cols:
        return pd.DataFrame(columns=['CoinA', 'CoinB'])

    coinA = [p.split('-')[0] for p in selected_cols]
    coinB = [p.split('-')[1] for p in selected_cols]
    selected_pairs = pd.DataFrame({'CoinA': coinA, 'CoinB': coinB})
    return selected_pairs

def makeCoinPairs(best_pairs):
    
    coinA = [p.split('-')[0] for p in best_pairs.index[0]]
    coinB = [p.split('-')[1] for p in best_pairs.index[0]]

    coins = pd.DataFrame({'CoinA': coinA, 'CoinB': coinB})

    return coins



def trainSymbols(candles: pd.DataFrame, highCorr_500: pd.DataFrame, max_pairs, step, r2=15, train_rows_start = 1000, train_rows_end = 2200):
    
    if max_pairs != 'all':
        symbols = highCorr_500['CoinA'].sample(max_pairs).unique().tolist()
        print('Total Amount of Unique Symbols:', len(symbols))
        
    else:
        symbols = highCorr_500['CoinA'].unique().tolist()
        print('Total Amount of Unique Symbols:', len(symbols))

    dfs = []
    for _,symbol in zip(tqdm(range(len(symbols))), symbols):
        
        
        
        
        while True:
            traincandles = candles[train_rows_start:train_rows_start + step]
            testcandles = candles[train_rows_start + step: train_rows_start + step + step]
            testcandles.reset_index(inplace = True)

            testcandles = testcandles.rename(columns = {'date':'time'})
            

            try:
                
                #corrCointegrated = corrCointegrated[corrCointegrated.isin(candles.columns)].dropna()
                #corrCointegrated = cointegrate(traincandles, highCorr_, MAX_KEEP=len(candles.columns))
                #highCorr_ = getHighCorrSymbols(traincandles)
                
                highCorr_ = highCorr_500[highCorr_500['CoinA'] == symbol]
                
                signals = Signals(r2,traincandles, testcandles, highCorr_)
            except Exception as e:
                print(f"Error occurred: {e}")
                continue

            result = evaluate_signals(highCorr_, signals, max_hold=50)
            
            #print(result.profit.sum())
            
            dfs.append(result)
        
            if train_rows_start >= train_rows_end:
                break
            train_rows_start += step
        

            
        

        
        


        #pair = pd.concat(dfs).groupby(level=0).sum().sort_values(0, ascending=False)
        #best_pairs.append(pair)


    #print(dfs)

    df_pairs = pd.concat(dfs)#.groupby(level=0).sum().sort_values('profit', ascending=False)
    df_pairs.index = pd.MultiIndex.from_arrays([df_pairs.index.str.split('-').str[0], df_pairs.index.str.split('-').str[1]], names=['CoinA', 'CoinB'])
    df_pairs.to_csv('Output/best_pairs.csv')
    return df_pairs

def testSymbols(candles:pd.DataFrame, pairs:pd.DataFrame, max_hold=30, train_step=200, test_step=200, testStart=1200, testEnd=1800):
    
    profit_all = []
    profit = []
    train_step = 200
    test_step = 200
    r2 = 15
    
    for i in range(testStart, testEnd, train_step):
        traincandles = candles[i-train_step:i]
    
        testcandles = candles[i:i+test_step]
        testcandles.reset_index(inplace = True)
        testcandles = testcandles.rename(columns = {'date':'time'})
        
        #highCorr = getHighCorrSymbols(traincandles)
    
        try:
            #corr = getHighCorrSymbols(traincandles[coins.values.flatten()])
            corr = select_pairs(traincandles, pairs, max_std=0.1, min_profit=0)
            
            
            signals = Signals(r2,traincandles, testcandles, corr)
            result = evaluate_signals(corr, signals, max_hold)
            
            profit.append(result)
            
            #print(max(result['std']))

        except Exception as e:
            print(f"Error occurred: {e}")
            continue

            
    df_ = pd.concat(profit)
    
    
    print('--- Final Results ---', '\n')
    print('Total Profit:', df_.profit.sum())
    print('Mean Profit per iteration:', df_.groupby(level = 0).sum().profit.mean())
    print('Max Standard Deviation of All Pairs:', df_.groupby(level = 0).sum()['std'].max())
    print('Total Trades:', df_.total_trades.sum())
    print('Max Hold Time:', max_hold)
    

    


if __name__ == "__main__":
    import multiprocessing as mp
    
    @dataclass
    class CoinPair:
        CoinA: str
        CoinB: str
        
    mp.set_start_method("spawn", force=True)
    
    
    candles = pd.read_csv('Output/hist_candles_1H.csv', index_col=0, header=[0,1]).xs('close', axis=1, level=1)
    
    for i in range(1000, 8000, 1000):
        candles_ = candles.sample(30, axis = 1)

        highCorr_500 = getHighCorrSymbols(candles_[i-1000:i])
        #highCorr_500 = cointegrate(candles_[i-1000:i], highCorr, MAX_KEEP=10000)['CoinA CoinB'.split()][:30]
        
        #symbols = pd.read_csv('Output/high_corr_symbols.csv').sample(500)


        pairs = trainSymbols(candles_, highCorr_500, max_pairs='all', step=200, r2=15, train_rows_start=i-1000, train_rows_end=i)

        pairs = pd.read_csv('Output/best_pairs.csv', index_col=[0,1])
        
        #print(pairs)
        
        max_std = 0.1
        min_profit = 0
        

        #pairs = pairs.sort_values('profit', ascending=False).reset_index()
        #pairs = pairs[pairs['std'] < 0.1]
        
        
        #coins = pairs['CoinA CoinB'.split()][:10]

        #print(pairs)
        testSymbols(candles_, pairs, max_hold=15, train_step=200, test_step=200, testStart=i + 200, testEnd=i + 1200)

        


