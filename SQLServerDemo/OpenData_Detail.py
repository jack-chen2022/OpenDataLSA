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

##取得[醫療]類相關標題(動態網站用)(LIST來源未知)
url="https://data.gov.tw/api/front/dataset/list"
b={"bool":[{"fulltext":{"value":"醫療"}}],
    "filter":[],
    "page_num":1,
    "page_limit":50,
    "tids":[],
    "sort":"dataset_view_times_desc"
 }
response=requests.post(url,
    headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36"
    },json=b
    )
res = json.loads(response.text)
print(response.status_code)
print(type(res))
datasets=res["payload"]["search_result"]
for data in datasets:
    titles=data["title"]
    nids=data["nid"]
    views=data["dataset_view_times"]
    downloads=data["resource_download_times"]
    comments=data["comment_quantity"]
    print("[公開資料標題]:"+titles)
    print("[公開資料編號]:"+str(nids))
    print("[關注次數]:"+str(views))
    print("[下載次數]"+str(downloads))
    print("[留言筆數]"+str(comments))



