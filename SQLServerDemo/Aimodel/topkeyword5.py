import glob
import json
import pandas as pd
import sqlite3
# 使用 jieba 進行分詞
import jieba
import os
import re
from urllib.request import urlretrieve
import jieba.analyse
#TFIDF文字雲
import os
import pandas as pd
import codecs
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

class topkeyword5:
    
    def JiebaDictionary(self):
        big_dict_path = "Aidata/dict.txt.big"
        if not os.path.exists(big_dict_path):
            print("下載大型字典")
            url = "https://github.com/fxsjy/jieba/raw/master/extra_dict/dict.txt.big"
            urlretrieve(url, big_dict_path)
        jieba.set_dictionary(big_dict_path)
        # 需加入一些常用字彙
        pro_dict_path = "Aidata/profession_dic.txt"
        if os.path.exists(pro_dict_path):
            print("載入專業詞典")
            jieba.load_userdict(pro_dict_path)
        return "Jibea已完成載入"
    #JiebaDictionary()

    def OpenDatadbconnect(self,keywords):
        conn = sqlite3.connect('Data/CorpousData.db')
        #cursor=conn.execute('SELECT * FROM V_CorpousConctData')
        cursor=conn.execute('SELECT * FROM V_CorpousConctData WHERE O_catego="{}"'.format(keywords))
        rows=cursor.fetchall()
        ocids=[]
        categos=[]
        titles=[]
        contents=[]
        for row in rows:
            ocids.append(row[0])
            categos.append(row[1])
            titles.append(row[2])
            contents.append(row[3])
            df=pd.DataFrame({
                    "O_cid_guest":ocids,
                    "O_catego":categos,
                    "O_titles":titles,
                    "O_contents":contents,
                },columns=["O_cid_guest","O_catego","O_titles","O_contents"])
        #print(df)
        return df
    #OpenDatadbconnect()
    
    # 對表格的每一筆都做出轉換
    # 將網址 英數字 特殊符號清除掉
    def OpenDatadataclean(self,content):
        # 將標點符號去掉
        punct = set(u''':!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒﹔﹕﹖﹗﹚﹜﹞！），．：；？$｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻︽︿﹁﹃﹙﹛﹝（｛“‘-—_…~/ －＊➜■─★☆=@<>◉é''')
        content = re.sub(r'https?:\/\/.*[\r\n]*','', content)
        content = re.sub(r'[0-9a-zA-Z%=://&..~：%!]','', content)
        content = "".join(filter(lambda x: x not in punct, content))
        # 去掉換行符號
        content = (" ".join(jieba.cut(content))
                        .replace("\n", "")
                        .replace("\r", ""))
        return content

    #取得關鍵字詞
    def Topkey(self,contentcut):
      key=jieba.analyse.extract_tags(contentcut,topK=5,allowPOS=())
      #keys=(','.join(key))
      print(key)
      return key

    #文字雲
    def CloudWord(self,keywords):
            ##轉成文字雲
        filename='Aidata/TFIDFData_Tags{}.txt'.format(keywords)
        textfilename=os.path.abspath('Aidata/TFIDFData_Tags{}.txt'.format(keywords))
        if os.path.exists(textfilename):
            text = open(filename,'rt',encoding='UTF-8').read()
        else:
            text = open('Aidata/TFIDFData_Tags.txt','rt',encoding='UTF-8').read()
            print('檔案不存在')
        #設定停用字(排除常用詞、無特殊意義字詞)
        stopwords ={}.fromkeys(["資料","檔案","請問","下載","消費","統計","福利部","更新","無法","服務","查詢"])
        wc=WordCloud(font_path="Aidata/NotoSerifTC-Black.otf",background_color="white",max_words =2000,stopwords=stopwords)
        wc.generate(text)
        wcpngfile=os.path.abspath('Aidata/{}_cloudword.png'.format(keywords))
        if os.path.exists(wcpngfile):
            print('檔案已經存在')
        else:
            wc.to_file('Aidata/{}_cloudword.png'.format(keywords))

