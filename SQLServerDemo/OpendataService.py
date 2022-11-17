#儲存資料庫
import sqlite3
from sqlalchemy import create_engine
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Table,MetaData,Column  #結構對應類型
from sqlalchemy.dialects.mssql import NCHAR,NVARCHAR,VARCHAR #引用資料庫風格的欄位
from sqlalchemy.orm import mapper
#轉入資料庫
from dbconnect.CsvToMySQL import Datatomysql
from dbconnect.DataToSqlite import Datatosqlite
#TFIDF文字雲
import time
import os
import pandas as pd
import codecs
import numpy as np
import matplotlib.pyplot as plt
from Aimodel.topkeyword5 import topkeyword5
from wordcloud import WordCloud
#GensimLSA
from Aimodel.GensimLSA import LSAModel
#MVC設計模式(Model/View/Controller)
import flask
from flask import Flask
from flask import Response
from flask import Flask, request
import OpenDataQuery as od
import OpenDataQueryPage as odpage
import json
import urllib.parse
from SQLServerDemo import app
#建構Flask物件
app=flask.Flask(__name__)

def index():
    return app.render_template('home.html')

#連接上資料庫
@app.route("/mysqlconn")
def connectionSQL():
	#建立資料庫連接Mapping，並且設定連接集區數量
	db = create_engine("mysql+pymysql://root:密碼@localhost:3306/iiif",pool_size=16,pool_pre_ping=True)
	print(db)
	return "連接成功"

@app.route('/opendata/query/<keyword>/rawdata')
def searchData(keyword):
    df=od.dataFind(keyword)
    return "成功完成爬網"

@app.route('/opendata/querypage')
def searchPageData():
    keyword=request.args.get('keyword')
    page=request.args.get('page')
    df=odpage.dataFind(keyword,page)
    return "成功完成爬網"

@app.route('/opendata/tomysql')
def DataTomysql():
    ##Save To MySQL
    #csvtosql.Listtomysql()
    ToMysql=Datatomysql()
    ToMysql.Listtomysql()
    ToMysql.Custtomysql()
    ToMysql.Replytomysql()
    return "成功資料匯入MySQL"

@app.route('/opendata/tosqlite')
def DataTosqlite():
    ##Save To SqLite
    ToSqLite=Datatosqlite()
    ToSqLite.Listtosqllite()
    ToSqLite.Custtosqllite()
    ToSqLite.Replytosqllite()
    return "成功資料匯入SqLite"

@app.route('/opendata/getdatalist',methods={'GET'})
def GetDatalist():
    conn = sqlite3.connect('Data/CorpousData.db')
    cursor=conn.cursor()
    rows=conn.execute('select * from opendatalists')
    #rows=cursor.fetchall()
    data=list()
    for row in rows:
		#建構辭典
        rec={'nids':row[1],'catego':row[2],'titles':row[3],'agency':row[4],'views':row[5],'downloads':row[6],'comments':row[7]}
        data.append(rec)
    #轉換成JSON字串
    result=flask.json.dumps(data)
    return Response(result,mimetype='application/json')

@app.route('/opendata/getdatagust',methods={'GET'})
def GetDataguest():
    conn = sqlite3.connect('Data/CorpousData.db')
    cursor=conn.cursor()
    rows=conn.execute('select * from opendataguest')
    #rows=cursor.fetchall()
    data=list()
    for row in rows:
		#建構辭典
        rec={'nid_guest':row[1],'cid_guest':row[2],'titles_guest':row[3],'contents_guest':row[4]}
        data.append(rec)
    #轉換成JSON字串
    result=flask.json.dumps(data)
    return Response(result,mimetype='application/json')

@app.route('/opendata/getdatareply',methods={'GET'})
def GetDatareply():
    conn = sqlite3.connect('Data/CorpousData.db')
    cursor=conn.cursor()
    rows=conn.execute('select * from opendatareply')
    #rows=cursor.fetchall()
    data=list()
    for row in rows:
		#建構辭典
        rec={'nid_reply':row[1],'pid_reply':row[2],'titles_reply':row[3],'contents_reply':row[4]}
        data.append(rec)
    #轉換成JSON字串
    result=flask.json.dumps(data)
    return Response(result,mimetype='application/json')

@app.route('/opendata/getdatatfidf',methods={'GET'})
def GetDataTfidf():
    conn = sqlite3.connect('Data/CorpousData.db')
    cursor=conn.cursor()
    rows=conn.execute('select * from V_CorpousTFIDF')
    #rows=cursor.fetchall()
    data=list()
    for row in rows:
		#建構辭典
        rec={'O_nids':row[0],'O_cid_guest':row[1],'O_catego':row[2],'O_titles':row[3],'O_contents':row[4],'O_agency':row[5],'O_titles_url':row[8],'O_views':row[9],'O_downloads':row[10],'O_comments':row[11],'keys':row[12],'Tags':row[13]}
        data.append(rec)
    #轉換成JSON字串
    result=flask.json.dumps(data)
    return Response(result,mimetype='application/json')

@app.route('/opendata/keywordtop5',methods={'POST'})
def TopKeyWord():
    keywords=request.args.get('keyword')
    #取得搜尋關鍵字Top5
    Topk5=topkeyword5()
    #Topk5.JiebaDictionary(self)
    df=Topk5.OpenDatadbconnect(keywords)
    #print(df)
    df_out=pd.DataFrame(df,columns=["O_content_cut"])
    df_out["O_content_cut"] = df["O_contents"].apply(Topk5.OpenDatadataclean)
    #print('[斷詞後]')
    #print(df_out)
    df_out["Tags"]=df_out["O_content_cut"].apply(Topk5.Topkey)
    #print(df_out)
    txtfile=codecs.open('Aidata/TFIDFData_Tags{}.txt'.format(keywords),'w','utf-8')
    #Tags匯出TXT檔
    rawdata=df_out["Tags"]
    for data in rawdata:
        datas=(','.join(data))
        txtfile.write(datas)
    return "完成資料清洗"

@app.route('/opendata/wordcloud',methods={'POST'})
def CloudWord():
    keywords=request.args.get('keyword')
    Topk5=topkeyword5()
    Topk5.CloudWord(keywords)
    return "產出文字雲"

@app.route('/opendata/gensimlsa',methods={'POST'})
def GensimLSA():
    document=request.get_json()
    doc=document["document"]
    #doc=request.args.get('document')
    Semantic=LSAModel()
    Semantic.OpenDatadbconnect()
    Semantic.LoadCorpus()
    Semantic.CreateLsi()
    similerDoc=Semantic.SearchSimilar(doc)
    df=Semantic.OpenDatadbconnect()
    data=list()
    for nid in similerDoc:
        searchanswer=df.iloc[nid]
        #print(type(searchanswer))
        rec={'O_nids':str(searchanswer['O_nids']),'O_catego':str(searchanswer['O_catego']),'O_titles':str(searchanswer['O_titles']),'O_titles_url':str(searchanswer['O_titles_url'])}
        data.append(rec)
    #轉換成JSON字串
    result=flask.json.dumps(data)
    #return "分析出五則相關文章"
    return Response(result,mimetype='application/json')
#主程式
if __name__=='__main__':
    #執行web Server
    app.run(host='localhost',port=8080)
