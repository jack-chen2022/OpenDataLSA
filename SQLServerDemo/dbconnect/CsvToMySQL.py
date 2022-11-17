from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Table,MetaData,Column  #結構對應類型
from sqlalchemy.dialects.mssql import NCHAR,NVARCHAR,VARCHAR,FLOAT,INTEGER #引用資料庫風格的欄位
from sqlalchemy.orm import mapper
import pymysql
import pandas as pd

class Datatomysql:
	def Listtomysql(self):
		#Step1 建立資料庫連接Mapping，並且設定連接集區數量
		#LocalDB MySQL
		engine = create_engine("mysql+pymysql://root:密碼@localhost:3306/iiif", echo=False)
		#Step2 Write DataFram To MySQL
		df=pd.read_csv('D:/data/resultdictDataList.csv',header=0,encoding='utf-8')
		#Step3 創建數據類型
		dtypedict={
			'inde':INTEGER(),
			'O_nids':INTEGER(),
			'O_catego':NVARCHAR(length=1024),
			'O_titles':NVARCHAR(length=1024),
			'O_contents':NVARCHAR(length=1024),
			'O_agency':NVARCHAR(length=1024),
			'O_views_num':INTEGER(),
			'O_downloads_num':INTEGER(),
			'O_comments_num':INTEGER(),
			'O_titles_url':NVARCHAR(length=1024)
		}
		#Step4 Write To MySQL
		df.to_sql(name='opendatalists',con=engine,if_exists='append',index=False,dtype=dtypedict)
		return "成功資料匯入MySQL"

	def Custtomysql(self):
		engine = create_engine("mysql+pymysql://root:MySQL!QAZ2wsx@localhost:3306/iiif", echo=False)
		df=pd.read_csv('D:/data/resultdictDataCust.csv',header=0,encoding='utf-8')
		dtypedict={
			'inde':INTEGER(),
			'O_nid_guest':INTEGER(),
			'O_cid_guest':INTEGER(),
			'O_titles_guest':NVARCHAR(length=1024),
			'O_contents_guest':NVARCHAR(length=1024)
		}
		df.to_sql(name='opendataguest',con=engine,if_exists='append',index=False,dtype=dtypedict)
		return "成功資料匯入MySQL"

	def Replytomysql(self):
		engine = create_engine("mysql+pymysql://root:MySQL!QAZ2wsx@localhost:3306/iiif", echo=False)
		df=pd.read_csv('D:/data/resultdictDataReply.csv',header=0,encoding='utf-8')
		dtypedict={
			'inde':INTEGER(),
			'O_nid_reply':INTEGER(),
			'O_pid_reply':INTEGER(),
			'O_titles_reply':NVARCHAR(length=1024),
			'O_contents_reply':NVARCHAR(length=1024)
		}
		df.to_sql(name='opendatareply',con=engine,if_exists='append',index=False,dtype=dtypedict)
		return "成功資料匯入MySQL"

#Listtomysql()


