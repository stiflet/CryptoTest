from bs4 import BeautifulSoup
import pandas as pd
from curl_cffi import AsyncSession, requests
import asyncio
from io import StringIO
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio  # ships with tqdm>=4.64
from threading import Thread

def run_async(func):
    def wrapper(*args,**kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper


def compareSymbols(zlimits_path, InvestingSymbols_path) -> pd.DataFrame:
    linkSymbols_df = pd.read_csv(InvestingSymbols_path)
    zlims_df = pd.read_csv(zlimits_path)

    zlims_unique = zlims_df[['coinA', 'coinB']].stack().map(lambda x: x.replace('USDT', '')).unique().tolist()
    symbols = linkSymbols_df[linkSymbols_df['Symbol'].isin(zlims_unique)]
    
    return symbols

def getTable(response, Symbol, dfs):
    html = response.text
    dataframes = pd.read_html(StringIO(html))
    labels = ['Moving Averages:', 'Technical Indicators:']


    for df in dataframes:
        if all(df.iloc[:,0].isin(labels)):
            df.columns = 'label verdict buy sell'.split()
            break

    try:

        df['buy'] = df['buy'].apply(lambda x: x.replace('Buy:', '').replace('(', '').replace(')',''))
        df['sell'] = df['sell'].apply(lambda x: x.replace('Sell:', '').replace('(', '').replace(')',''))
    
        df.set_index('label', inplace = True)
        df = df.T
        df.columns = pd.MultiIndex.from_product([[Symbol], df.columns])
        dfs.append(df)
    
    except Exception as e:
        print(f"Error occurred while processing {Symbol}: {e}")


async def getData(LinkSymbol, ses: AsyncSession):
    retries = 0

    while retries <= 5:
        response = await ses.get(f'https://www.investing.com/crypto/{LinkSymbol}/technical', impersonate = 'chrome110')
        if response.status_code == 200:
            break
        retries += 1

    return response
    

@run_async
async def getTables(zlimits_path, InvestingSymbols_path = r'Final\Investing.com\Data\AllSymbols.csv'):
    dfs = []
    tasks = []
    threads = []
    symbols = compareSymbols(zlimits_path, InvestingSymbols_path)
    async with AsyncSession() as ses:
        for symbol in symbols['LinkSymbol'].tolist():
            tasks.append(getData(symbol,ses))
        responses = await tqdm_asyncio.gather(*tasks, desc = 'Fetching HTML')
        
    for response, symbol, _ in zip(responses, symbols['Symbol'].tolist(), tqdm(range(len(responses)), desc = 'Creating Table')):
        threads.append(Thread(target = getTable, args = (response, symbol, dfs)))
    for t in threads:
        try:
            t.start()
            t.join()
        except Exception as e:
            print(f"Error occurred while processing {symbol}: {e}")
            continue
    df = pd.concat(dfs, axis = 1).stack(future_stack = True)
    df.to_csv(r'Output/technical_indicators.csv')
    return dfs


if __name__ == '__main__':
    InvestingSymbols_path = r'Investing.com/Data/AllSymbols.csv'
    zlimits_path = r'Output/zlimits.csv'
    dfs = getTables(zlimits_path, InvestingSymbols_path)

