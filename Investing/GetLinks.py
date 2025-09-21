from camoufox.sync_api import Camoufox
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import pandas as pd

def parseHTML(content):
    soup = BeautifulSoup(content, features = 'lxml')
    row_elements = soup.select('tr.datatable_row__Hk3IV td:nth-child(4)')
    linkSymbols = []
    Symbols = []
    for td in row_elements:
        try:
            LinkSymbol = td.select_one('div > a').get('href').split('/')[-1]
            Symbol = td.select_one('div span:nth-child(2)').getText()
            linkSymbols.append(LinkSymbol)
            Symbols.append(Symbol)
        except:
            continue
    df = pd.DataFrame({'Symbol':Symbols, 'LinkSymbol':linkSymbols})
    return df



def getHtml():
    with Camoufox(
        headless = True) as browser:
        page = browser.new_page()

        page.goto('https://www.investing.com/crypto/currencies')
        page.get_by_role('button', name='I Accept').click()
        loadMore = page.get_by_text(text = 'Load more')
        signInBox = page.locator('div svg[data-test="sign-up-dialog-close-button"]')
        
        clickTime = datetime.now()
        while True:
            if signInBox.is_visible():
                signInBox.click()
                
            if loadMore.is_visible():
                loadMore.click()
                clickTime = datetime.now()
            
            elif clickTime + timedelta(seconds = 10) < datetime.now():
                break
        content = page.content()
        df = parseHTML(content)
        df.to_csv('Final/Investing.com/AllSymbols.csv', index = False)
        print(df)
        
        
if __name__ == '__main__':
    getHtml()