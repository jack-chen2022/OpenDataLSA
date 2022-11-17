from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.error import HTTPError
from jieba.analyse import extract_tags
import requests
import json
import time
import pandas as pd
#import selenium.webdriver.support.ui as ui
#from selenium import webdriver
#from selenium.webdriver.common.by import By
#from webdriver_manager.chrome import ChromeDriverManager
#from selenium.webdriver import Chrome
#from prettytable import prettytable

#取得各標題內容/客戶反映
url="https://data.gov.tw/api/front/comment/detail"
response=requests.post(url,
    headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36"
    },data={"nid":"6624"})
res = json.loads(response.text)
#p1 = prettytable.PrettyTable(["cid","nid","title", "body"], encodings="utf-8") 
datasets_1=res["payload"]["results"]
for data in datasets_1:
    titles=data["title"]
    bodys=data["body"]
    print("訪客留言標題:"+titles)
    print("訪客留言內容:"+bodys)
    #print("------------")
    datasets_2=data["reply"]
    for data2 in datasets_2:
        titles_reply=data2["title"]
        bodys_reply=data2["body"]
        print("官方回覆標題:"+titles_reply)
        print("官方回覆內容:"+bodys_reply)
        print("------------")


