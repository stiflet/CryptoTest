from curl_cffi.requests import AsyncSession
from curl_cffi import requests
import pandas as pd
import asyncio
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from tqdm.asyncio import tqdm_asyncio

os.makedirs('Output', exist_ok=True)
def runAsync(function):
    def wrapper(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))
    return wrapper

async def getData(session, symbol, gran, startTime, endTime, semaphore, limit=200):
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
    df = pd.concat(dfs, axis = 1)
    
    if save:
        df.to_csv(f'Output/hist_candles_{gran}.csv')
    return df

if __name__ == '__main__':

    #symbols = pd.read_csv('Output/zlimSymbols.csv')['CoinA CoinB'.split()].stack().unique().tolist()
    symbols = pd.read_csv(r'PrepareData/Data/high_corr_symbols.csv').stack().unique().tolist()
    getHistCandles(symbols, gran='1H', loops=50)
