
import pandas as pd
import pytrends

from pytrends.request import TrendReq
pytrend = TrendReq()


KEYWORDS=['Nike','Adidas','Under Armour','Zara','H&M','Louis Vuitton'] 
KEYWORDS_CODES=[pytrend.suggestions(keyword=i)[0] for i in KEYWORDS] 
df_CODES= pd.DataFrame(KEYWORDS_CODES)
df_CODES

EXACT_KEYWORDS=df_CODES['mid'].to_list()
DATE_INTERVAL='2020-01-01 2020-05-01'
COUNTRY=["US","GB","DE"] #Use this link for iso country code
CATEGORY=0 # Use this link to select categories
SEARCH_TYPE='' #default is 'web searches',others include 'images','news','youtube','froogle' (google shopping)

Individual_EXACT_KEYWORD = list(zip(*[iter(EXACT_KEYWORDS)]*1))
Individual_EXACT_KEYWORD = [list(x) for x in Individual_EXACT_KEYWORD]
dicti = {}
i = 1
for Country in COUNTRY:
    for keyword in Individual_EXACT_KEYWORD:
        pytrend.build_payload(kw_list=keyword, 
                              timeframe = DATE_INTERVAL, 
                              geo = Country, 
                              cat=CATEGORY,
                              gprop=SEARCH_TYPE) 
        dicti[i] = pytrend.interest_over_time()
        i+=1
df_trends = pd.concat(dicti, axis=1)



df_trends.columns = df_trends.columns.droplevel(0) #drop outside header
df_trends = df_trends.drop('isPartial', axis = 1) #drop "isPartial"
df_trends.reset_index(level=0,inplace=True) #reset_index
df_trends.columns=['date','Nike-US','Adidas-US','Under Armour-US','Zara-US','H&M-US','Louis Vuitton-US','Nike-UK','Adidas-UK','Under Armour-UK','Zara-UK','H&M-UK','Louis Vuitton-UK',
'Nike-Germany','Adidas-Germany','Under Armour-Germany','Zara-Germany','H&M-Germany','Louis Vuitton-Germany'] #change column names


