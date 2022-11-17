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
searchName='醫療'
pagenum=5
while True:
    b={"bool":[{"fulltext":{"value":searchName}}],
    "filter":[],
    "page_num":pagenum,
    "page_limit":50,
    "tids":[],
    "sort":"dataset_view_times_desc"}
    try:
        response=requests.post(url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36"
            },json=b)
    except HTTPError:
        print("已達最後一頁")
        break
    #print("首頁連結:"+str(response.status_code))
    res = json.loads(response.text)
    datasets=res["payload"]["search_result"]
    for data in datasets:
        titles=data["title"]
        nids=data["nid"]
        category=data["category_name"]
        agency=data["agency_name"]
        views=data["dataset_view_times"]
        downloads=data["resource_download_times"]
        comments=data["comment_quantity"]
        print("[公開資料標題]:"+titles)
        print("[公開資料編號]:"+str(nids))
        print("[提供機關]:"+agency)
        print("[服務分類]"+category)
        print("[關注次數]:"+str(views))
        print("[下載次數]"+str(downloads))
        print("[留言筆數]"+str(comments))
    #取得各標題內容/客戶反映
        url2="https://data.gov.tw/api/front/comment/detail"
        response2=requests.post(url2,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36"
        },data={"nid":str(nids)})

        print("HTTP狀態碼"+str(response2.status_code))
        res2 = json.loads(response2.text)
        datasets_1=res2["payload"]["results"]
        for data in datasets_1:
            titles=data["title"]
            bodys=data["body"]
            nid=data["nid"]
            print("[公開資料編號]"+str(nid))
            print("[訪客留言標題]:"+titles)
            print("[訪客留言內容]:"+bodys)
            datasets_2=data["reply"]
            for data2 in datasets_2:
                titles_reply=data2["title"]
                bodys_reply=data2["body"]
                print("[官方回覆標題]:"+titles_reply)
                print("[官方回覆內容]:"+bodys_reply)
            print("------------")
            time.sleep(1)
    pagenum=pagenum+1        
     