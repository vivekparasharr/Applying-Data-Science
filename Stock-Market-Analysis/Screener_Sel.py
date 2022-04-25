# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 01:35:18 2020

@author: Raj Verma
"""

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import lxml.html as lh
import pandas as pd
file_data = []
df1=pd.DataFrame()



from selenium import webdriver
driver = webdriver.Chrome()
login = "https://www.screener.in/login/"
driver.get(login)
username = driver.find_element_by_id("id_username")
username.clear()
username.send_keys("verma-raj@hotmail.com")
password = driver.find_element_by_name("password")
password.clear()
password.send_keys("Prince420$")
driver.find_element_by_xpath("/html/body/main/div/div/div[2]/form/button").click()
url1="https://www.screener.in/screen/raw/?sort=&order=&source=&query=Market+capitalization+%3E+5%0D%0A&page="
Limit="&limit=100"

n=3
for pg in range(1, n):
    URL=url1+str(pg)+Limit
    page = driver.get(URL)
    body_elements = driver.find_element_by_xpath('//tbody').get_attribute("innerHTML")
    doc = lh.fromstring(body_elements)
    tr_elements = doc.xpath('//tr')
    #Create empty list
    col=[]
    i=0
    #For each row, store each first element (header) and an empty list
    for t in tr_elements[0]:
        i+=1
        name=t.text_content()
        print ('%d:"%s"'%(i,name))
        col.append((name,[]))
    #Since out first row is the header, data is stored on the second row onwards
        for j in range(1,len(tr_elements)):
            #T is our j'th row
            T=tr_elements[j]
        
            #If row is not of size 10, the //tr data is not from our table 
            if len(T)!=11:
                break
        
            #i is the index of our column
            i=0
        
            #Iterate through each element of the row
            for t in T.iterchildren():
                data=t.text_content() 
                #Check if row is empty
                if i>0:
                    #Convert any numerical value to integers
                    try:
                        data=int(data)
                    except:
                        pass
                        #Append the data to the empty list of the i'th column
                        col[i][1].append(data)
                        #Increment i for the next column
                        i+=1
                        Dict={title:column for (title,column) in col}
                        df=pd.DataFrame(Dict)
                        df1=df.append(df1)
                   
                    
import pandas as pd 
df = pd.read_html('https://en.wikipedia.org/wiki/Demographics_of_Chile')
df[1]
df[2]

df1 = pd.DataFrame({'a':[1,2,3],'b':[11,12,13]})
df2 = pd.DataFrame({'a':[4,5,6],'b':[14,15,16]})
df1.append(df2)
