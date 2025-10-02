from curl_cffi.requests import AsyncSession
from curl_cffi import requests
import pandas as pd
import asyncio
import os
import numpy as np
from datetime import datetime, timedelta
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

os.makedirs('Output', exist_ok=True)
def runAsync(function):
    def wrapper(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))
    return wrapper


def getSymbols(minVol: int, save:bool = False):
    url = 'https://api.bitget.com/api/v2/mix/market/tickers?productType=USDT-FUTURES'
    response = requests.get(url)
    df = pd.DataFrame(response.json()['data'])
    df = df[pd.to_numeric(df['usdtVolume']) > minVol]
    df = df['symbol'].tolist()
    if save:
        
        df.to_csv('Output/symbols.csv', index=False)
        
    return df


async def getData(session, symbol, gran, startTime, endTime, semaphore, limit=200) -> pd.DataFrame:
    url = f'https://api.bitget.com/api/v2/mix/market/history-candles?symbol={symbol}&granularity={gran}&limit={limit}&productType=usdt-futures&'+ \
        f'startTime={int(startTime.timestamp() * 1000)}&endTime={int(endTime.timestamp() * 1000)}'
    async with semaphore:
        retries = 0
        while retries < 5:
            response = await session.get(url)
            if response.status_code != 200:
                retries += 1
                await asyncio.sleep(2)
                continue
            break
    return pd.DataFrame(response.json()['data'])

async def runLoop_hours(symbol, gran, semaphore, loops, separate: bool = False) -> pd.DataFrame:
    async with AsyncSession() as session:
        tasks = []
        endTime = datetime.now()
        startTime = endTime - timedelta(hours=200)
        for _ in range(loops):
            tasks.append(getData(session, symbol, gran, startTime, endTime, semaphore))
            endTime = startTime
            startTime = endTime - timedelta(hours=200)
            
        dfs = await asyncio.gather(*tasks)

    df = pd.concat(dfs, axis = 0)
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'volumeQuote']
    df = df.apply(pd.to_numeric)
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df.set_index('date', inplace = True)
    df.sort_values('date', inplace = True)
    df.columns = pd.MultiIndex.from_product([[symbol], df.columns])
    
    if separate:
        df.to_csv(f'Output/{symbol}_{gran}.csv', index = False)
    return df

async def runLoop_days(symbol, gran, semaphore, loops, separate: bool = False) -> pd.DataFrame:
    async with AsyncSession() as session:
        tasks = []
        endTime = datetime.now()
        startTime = endTime - timedelta(days=50)
        for _ in range(loops):
            tasks.append(getData(session, symbol, gran, startTime, endTime, semaphore))
            endTime = startTime
            startTime = endTime - timedelta(days=50)

        dfs = await asyncio.gather(*tasks)
        
    try:
        df = pd.concat(dfs, axis = 0)
    except:
        df = pd.DataFrame()
    
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'volumeQuote']
    df = df.apply(pd.to_numeric)
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df.set_index('date', inplace = True)
    df.sort_values('date', inplace = True)
    
    df.columns = pd.MultiIndex.from_product([[symbol], df.columns])
    
    if separate:
        df.to_csv(f'Output/{symbol}_{gran}.csv', index = False)
    return df

@runAsync
async def getHistCandles(symbols, gran, loops=5, separate: bool = False, save: bool = True) -> pd.DataFrame:
    semaphore = asyncio.Semaphore(5)
    if 'H' in gran:
        tasks = [runLoop_hours(symbol, gran=gran, semaphore=semaphore, loops=loops, separate=separate) for symbol in symbols]
    elif 'D' in gran:
        tasks = [runLoop_days(symbol, gran=gran, semaphore=semaphore, loops=loops, separate=separate) for symbol in symbols]
    dfs = await tqdm_asyncio.gather(*tasks, desc = 'Fetching Price Data')
    df = pd.concat(dfs, axis = 1).dropna(how = 'any', axis = 1)
    
    if save:
        df.to_csv(f'Output/hist_candles_{gran}.csv')
    return df


class Load():
    def __init__(self, symbols, gran, loops, separate = False, save = False):
        self.candles = getHistCandles(symbols, gran, loops, separate, save)
        
        
    def correlate(self, save: bool = False):
        dfCorr = self.candles.xs('close', 1, 1).corr()    
        np.fill_diagonal(dfCorr.values, 0)
        df = pd.DataFrame(dfCorr.where(dfCorr > 0.9).stack().index.tolist(), columns=['CoinA', 'CoinB'])
        symbols = df
        if save:
            symbols.to_csv('Output/high_corr_symbols.csv', index=False)

        self.corr = symbols
        return symbols
    
    def cointegrate(self, BASE_UNITS_A = None, BASE_NOTIONAL_A = 20.0, MAX_KEEP = 200, MIN_OBS = 50, PVAL_THRESH = 0.05):
        
        highCorr = self.correlate()
        candles = self.candles.xs('close', 1, 1)
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

        results_candles = pd.DataFrame(results)#.sort_values('p_value')
        results_candles.to_csv('Output/zlimSymbols.csv', index=False)

        self.coint = results_candles
        
        return results_candles


        
    
    def getZscores(self, method:str, rolling_spread = 5, rolling_spreadMean = 30) -> pd.DataFrame:
        """Build z-scores for each CoinA-CoinB column. Robust to NaNs."""
        candles = self.candles.xs('close', 1, 1)
        
        if method == 'correlate':
            highCorr = self.correlate()
        elif method == 'cointegrate':
            highCorr = self.cointegrate()

        else:
            raise Exception('Choose between: cointegrate or correlate')
        zscores = []
        for r in highCorr.itertuples():
            coinPair = candles[[r.CoinA, r.CoinB]].copy()
            spread = coinPair[r.CoinA] - coinPair[r.CoinB]
            spread_mean = spread.rolling(rolling_spread, min_periods=rolling_spread).mean()
            sigma = spread_mean.rolling(rolling_spreadMean, min_periods=rolling_spreadMean).std()
            zscore = (spread - spread_mean) / sigma
            col_name = f"{r.CoinA}-{r.CoinB}"
            zscores.append(zscore.rename(col_name))
    
        zscores_df = pd.concat(zscores, axis=1)

        self.zscores = zscores_df

        return zscores_df
    
if __name__ == '__main__':

    btc = Load(['BTCUSDT', 'ETHUSDT'], gran='1H', loops=2)
    print(btc.candles)
    
    print(btc.cointegrate())
