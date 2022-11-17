from urllib.request import urlopen
from urllib.error import HTTPError
from jieba.analyse import extract_tags
import requests
import json
import time
import pandas as pd
from dbconnect.CsvToMySQL import Datatomysql

##取得[醫療]類相關標題(動態網站用)(LIST來源未知)
url="https://data.gov.tw/api/front/dataset/list"
pagenum=5
searchName='醫療'
b={"bool":[{"fulltext":{"value":searchName}}],
    "filter":[],
    "page_num":pagenum,
    "page_limit":50,
    "tids":[],
    "sort":"dataset_view_times_desc"}
response=requests.post(url,
    headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36"
    },json=b)
res = json.loads(response.text)
datasets=res["payload"]["search_result"]
#print(len(datasets))
for data in datasets:
        nids_list=data["nid"]
        category_list=data["category_name"]
        titles_list=data["title"]
        agency_list=data["agency_name"]
        views_list=data["dataset_view_times"]
        downloads_list=data["resource_download_times"]
        comments_list=data["comment_quantity"]
        #print("[公開資料標題]:"+titles)
        #print("[公開資料編號]:"+str(nids))
        #print("[提供機關]:"+agency)
        #print("[服務分類]"+category)
        #print("[關注次數]:"+str(views))
        #print("[下載次數]"+str(downloads))
        #print("[留言筆數]"+str(comments))
        dictDataList={'nids':nids_list,'catego':category_list,'titles':titles_list,'agency':agency_list,'views':views_list,'downloads':downloads_list,'comments':comments_list}
        #封裝成df
        ##https://blog.csdn.net/weixin_39750084/article/details/81429037
        resultdictDataList=pd.DataFrame.from_dict(dictDataList,orient='index').T
        resultdictDataList.index.name='inde'
        #print(dictDataList)
        print(resultdictDataList)
        ##傳送DataToMySQL函數接收
        #datatomysql=Datatomysql()
        #datatomysql.Listtomysql(resultdictDataList)
        #取得各標題內容/客戶反映
        url2="https://data.gov.tw/api/front/comment/detail"
        response2=requests.post(url2,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36"
        },data={"nid":str(nids_list)})
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
