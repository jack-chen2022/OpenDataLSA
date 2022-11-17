import sqlite3
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Table,MetaData,Column  #結構對應類型
from sqlalchemy.dialects.mssql import NCHAR,NVARCHAR,VARCHAR,FLOAT,INTEGER #引用資料庫風格的欄位
from sqlalchemy.orm import mapper
import pymysql
import pandas as pd

class Datatosqlite:
	def Listtosqllite(self):
		#Create Table
		#cur.execute("CREATE TABLE movie(title, year, score)")
		#Insert Data
		#cur.execute("""INSERT INTO movie VALUES ('Monty Python and the Holy Grail', 1980, 8.2),('And Now for Something Completely Different', 1999, 7.5)""")
		#con.commit()
		#Select Data
		#res=cur.execute("SELECT name FROM sqlite_master")
		#res.fetchone()
		#Connect Sqlite
		conn=sqlite3.connect('Data/OpenData.db')
		cur=conn.cursor()
		cur.execute("""CREATE TABLE IF NOT EXISTS opendatalists(inde integer,O_nids integer,O_catego text,O_titles text,O_contents text,O_agency text,O_views_num integer,O_downloads_num integer,O_comments_num integer,O_titles_url text)""")
		df=pd.read_csv('D:/data/resultdictDataList.csv',header=0,encoding='utf-8')
		df.columns = df.columns.str.strip()
		df.to_sql(name='opendatalists',con=conn,if_exists='append',index=False)
		return "成功匯入Sqlite"

	def Custtosqllite(self):
		conn=sqlite3.connect('Data/OpenData.db')
		cur=conn.cursor()
		cur.execute("""CREATE TABLE IF NOT EXISTS opendataguest(inde integer,O_nid_guest integer,O_cid_guest integer,O_titles_guest text,O_contents_guest text)""")
		df=pd.read_csv('D:/data/resultdictDataCust.csv',header=0,encoding='utf-8')
		df.columns = df.columns.str.strip()
		df.to_sql(name='opendataguest',con=conn,if_exists='append',index=False)
		return "成功匯入Sqlite"

	def Replytosqllite(self):
		conn=sqlite3.connect('Data/OpenData.db')
		cur=conn.cursor()
		cur.execute("""CREATE TABLE IF NOT EXISTS opendatareply(inde integer,O_nid_reply integer,O_pid_reply integer,O_titles_reply text,O_contents_reply text)""")
		df=pd.read_csv('D:/data/resultdictDataReply.csv',header=0,encoding='utf-8')
		df.columns = df.columns.str.strip()
		df.to_sql(name='opendatareply',con=conn,if_exists='append',index=False)
		return "成功匯入Sqlite"
	#Listtosqllite()
