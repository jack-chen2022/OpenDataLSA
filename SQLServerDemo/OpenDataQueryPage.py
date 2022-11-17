#from bs4 import BeautifulSoup
#from jieba.analyse import extract_tags
from urllib.request import urlopen
from urllib.error import HTTPError
import requests
import json
import time
import pandas as pd

###取得[醫療]類相關標題(動態網站用)
def dataFind(keyword,page):
    #searchName='運動'
    #searchName=keyword
    #pagenum=int(page)
    #pagenum=5
    #回傳JSON
    titles_list=list() #1.公開資料標題
    nids_list=list()   #2.公開資料編號
    agency_list=list() #3.提供機關
    category_list=list() #4.服務分類
    views_list=list()  #5.關注次數
    downloads_list=list()  #6.下載次數
    comments_list=list() #7.留言筆數
    url_list=list()    #8.留言網址
    contents_list=list()    #9.內容說明
    nid_custreply_list=list() #1.公開資料編號
    cid_custreply_list=list()#2.留言編號
    custitles_list=list() #3.訪客留言標題
    cusbodys_list=list()  #4.訪客留言內容
    nid_govreply_list=list() #1.公開資料編號
    pid_govreply_list=list() #2.留言編號
    titles_reply_list=list() #3.官方回覆標題
    bodys_reply_list=list()  #4.官方回覆內容
    #print("已到第{}頁:".format(pagenum))

    url="https://data.gov.tw/api/front/dataset/list"
    #pagenum=2
    #searchName='運動'
    pagenum=int(page)
    searchName=str(keyword)
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


    for data in datasets:
            print('===開放資料標題===')
            nids=data["nid"]
            category=data["category_name"]
            titles=data["title"]
            agency=data["agency_name"]
            views=data["dataset_view_times"]
            downloads=data["resource_download_times"]
            comments=data["comment_quantity"]
            contents=data["content"]
            urls='https://data.gov.tw/dataset/'+str(nids)
            titles_list.append(titles) #1.公開資料標題
            nids_list.append(nids) #2.公開資料編號
            agency_list.append(agency) #3.提供機關
            category_list.append(category) #4.服務分類
            views_list.append(views) #5.關注次數
            downloads_list.append(downloads) #6.下載次數
            comments_list.append(comments) #7.留言筆數
            url_list.append(urls)    #8.留言網址
            contents_list.append(contents) #9.內容說明
            #Scraping_time
            dictDataList={'nids':nids_list,'catego':category_list,'titles':titles_list,'contents':contents_list,'agency':agency_list,'views':views_list,'downloads':downloads_list,'comments':comments_list,'urls':url_list}
            #封裝成df
            resultdictDataList=pd.DataFrame(dictDataList)
            resultdictDataList.index.name='inde'
            #print(dictDataList)
            print(resultdictDataList)
            #取得各標題-客戶反映內容
            url2="https://data.gov.tw/api/front/comment/detail"
            response2=requests.post(url2,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36"
            },data={"nid":str(nids)})
            print("HTTP狀態碼"+str(response2.status_code))
            res2 = json.loads(response2.text)
            datasets_1=res2["payload"]["results"]
            #客戶反映
            for data in datasets_1:
                print('===問題留言===')
                nid_cust=data["nid"]
                cid_cust=data["cid"]
                custitles=data["title"]
                cusbodys=data["body"]
                nid_custreply_list.append(nid_cust) #1.公開資料編號
                cid_custreply_list.append(cid_cust) #2.留言編號
                custitles_list.append(custitles) #3.訪客留言標題
                cusbodys_list.append(cusbodys)  #4.訪客留言內容
                #Scraping_time
                dictDataCust={'nid_reply':nid_custreply_list,'cid_reply':cid_custreply_list,'titles_cus':custitles_list,'bodys_cus':cusbodys_list}
                #封裝成df
                resultdictDataCust=pd.DataFrame(dictDataCust)
                resultdictDataCust.index.name='inde'
                print(dictDataCust)
                print(resultdictDataCust)
                ##Save To Json 
                resultdictDataCust.to_json('D:/data/resultdictDataCust{}.json'.format(nid_cust),orient = 'records')
                #官方答覆
                print('===官方回應===')
                datasets_2=data["reply"]
                for data2 in datasets_2:
                    nid_gov=data2["nid"]
                    pid_gov=data2["pid"]
                    titles_reply=data2["title"]
                    bodys_reply=data2["body"]
                    nid_govreply_list.append(nid_gov) #1.公開資料編號
                    pid_govreply_list.append(pid_gov) #2.留言編號
                    titles_reply_list.append(titles_reply) #3.官方回覆標題
                    bodys_reply_list.append(bodys_reply)  #4.官方回覆內容
                    #Scraping_time
                    dictDataReply={'nid_reply':nid_govreply_list,'pid_reply':pid_govreply_list,'titles_reply':titles_reply_list,'bodys_reply':bodys_reply_list}
                    #封裝成df
                    resultdictDataReply=pd.DataFrame(dictDataReply)
                    resultdictDataReply.index.name='inde'
                    print(dictDataReply)
                    print(resultdictDataReply)
                    #Save To Json
                    #resultdictDataReply.to_json('D:/data/resultdictDataReply{}.json'.format(nid_cust),orient = 'records')
            time.sleep(1)
    #pagenum=pagenum+1
    data1=resultdictDataList
    data2=resultdictDataCust
    #data3=resultdictDataReply
    ##Data Clean
    #Save To Json
    resultdictDataList.to_json('D:/data/resultdictDataList.json',orient = 'records')
    ##Save To CSV 
    resultdictDataList.to_csv('D:/data/resultdictDataList.csv',encoding='utf-8-sig')
    resultdictDataCust.to_csv('D:/data/resultdictDataCust.csv',encoding='utf-8-sig')
    resultdictDataReply.to_csv('D:/data/resultdictDataReply.csv',encoding='utf-8-sig')

    ##客戶各留言代碼cid <=> 官方回應留言代碼pid
    ##查詢頁標題代碼nid <=> 對應留言代碼nid

     