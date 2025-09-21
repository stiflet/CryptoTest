import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from tqdm import tqdm


def getHighCorrSymbols(candles: pd.DataFrame, save: bool = False):
    dfCorr = candles.corr()    
    np.fill_diagonal(dfCorr.values, 0)
    symbols = pd.DataFrame(dfCorr.where(dfCorr > 0.9).stack().index.tolist(), columns=['CoinA', 'CoinB'])
    
    if save:
        symbols.to_csv('Output/high_corr_symbols.csv', index=False)    
    return symbols


def cointegrate(candles, highCorr, BASE_UNITS_A = None, BASE_NOTIONAL_A = 20.0, MAX_KEEP = 200, MIN_OBS = 50, PVAL_THRESH = 0.05):
    results = []
    pbar = tqdm(total=MAX_KEEP, desc="Processing zlimits pairs")
    for r in highCorr.itertuples(index=False):
        sA = candles[r.CoinA].astype(float)
        sB = candles[r.CoinB].astype(float)

        pair = pd.concat([sA, sB], axis=1, keys=['A','B']).dropna()
        if len(pair) < MIN_OBS:
            continue

        # Engle–Granger: test A vs B
        t_stat, p_val, crit = coint(pair['A'], pair['B'])
        if p_val >= PVAL_THRESH:
            continue

        # OLS: A ~ alpha + beta * B
        X = sm.add_constant(pair['B'])
        ols = sm.OLS(pair['A'], X).fit()
        alpha = ols.params.iloc[0]
        beta  = ols.params.iloc[1]


        # Latest prices for sizing
        try:
            px_A = pair['A'].iloc[-1]
            px_B = pair['B'].iloc[-1]
            
        except Exception as e:
            print(f"Error occurred: {e}")
            continue

        # ----- SIZING --------------------------------------------------------
        # Beta-neutral: q_B = beta * q_A  (this matches the spread A - beta*B)
        if BASE_UNITS_A is not None:
            qty_A_units = BASE_UNITS_A
        else:
            # Use notional for CoinA to determine units
            qty_A_units = BASE_NOTIONAL_A / px_A

        qty_B_units_beta = beta * qty_A_units
        
        notional_A_usdt      = qty_A_units       * px_A
        notional_B_usdt_beta = qty_B_units_beta  * px_B

        # Dollar-neutral alternative (ignores beta; sets |$A| = |$B|)
        qty_B_units_dollar_neutral = notional_A_usdt / px_B

        spread = pair['A'] - (alpha + beta * pair['B'])

        results.append({
            'CoinA': r.CoinA,
            'CoinB': r.CoinB,
            'beta': beta,
            'qty_A_units': float(qty_A_units),
            'qty_B_units_beta': float(qty_B_units_beta),
            'notional_A_usdt': float(notional_A_usdt),
            'notional_B_usdt_beta': float(notional_B_usdt_beta),
            'qty_B_units_dollar_neutral': float(qty_B_units_dollar_neutral),
            'p_value': float(p_val)
        })
        
        pbar.update(1)

        if len(results) >= MAX_KEEP:
            break

    results_candles = pd.DataFrame(results)
    results_candles.to_csv('Output/zlimSymbols.csv', index=False)
    
    return results_candles

    
    
    