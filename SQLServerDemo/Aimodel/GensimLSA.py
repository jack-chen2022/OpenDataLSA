import sqlite3
import json
import pandas as pd
#import glob
#import matplotlib.pyplot as plt
#import numpy as np
import jieba.analyse
import jieba
import codecs
import os
import re
#LSA Semantic
from gensim import corpora, models, similarities
import gensim.downloader as api
from gensim.models import TfidfModel
from gensim.corpora import Dictionary

class LSAModel:
    def OpenDatadbconnect(self):
        global df
        conn = sqlite3.connect("Data/CorpousData.db")
        cursor=conn.execute('SELECT * FROM V_CorpousWord2Vec')
        rows=cursor.fetchall()
        O_nids=[]
        O_cid_guest=[]
        O_catego=[]
        O_titles=[]
        O_titles_guest=[]
        O_content_cut=[]
        O_titles_url=[]
        keys=[]
        Tags=[]
        for row in rows:
            O_nids.append(row[0])
            O_cid_guest.append(row[1])
            O_catego.append(row[2])
            O_titles.append(row[3])
            O_titles_guest.append(row[4])
            O_content_cut.append(row[5])
            O_titles_url.append(row[6])
            keys.append(row[7])
            Tags.append(row[8])
        df=pd.DataFrame({
              "O_nids":O_nids,
              "O_cid_guest":O_cid_guest,
              "O_catego":O_catego,
              "O_titles":O_titles,
              "O_titles_guest":O_titles_guest,
              "O_content_cut":O_content_cut,
              "O_titles_url":O_titles_url,
              "keys":keys,
              "Tags":Tags,
            },columns=["O_nids","O_cid_guest","O_catego","O_titles","O_titles_guest","O_content_cut","O_titles_url","keys","Tags"])
        print(df)
        return df


    def LoadCorpus(self):
        # 載入語料庫(將產生的Index放入IIF資料夾)
        global dictionary,corpus
        if (os.path.exists('Aidata/CorpousData_contentcut.dict')):
            dictionary = corpora.Dictionary.load('Aidata/CorpousData_contentcut.dict')
            corpus = corpora.MmCorpus('Aidata/CorpousData_contentcut.mm') # 將數據流的語料變為內容流的語料
            print("Used files generated from dictionary and corpus")
        else:
            print("Please run first tutorial to generate data set")


    def CreateLsi(self):
        global corpus_lsi,lsi
        # 創建 tfidf model
        tfidf = models.TfidfModel(corpus)
        # 轉為向量表示
        corpus_tfidf = tfidf[corpus]
        # 創建 LSI model 潛在語義索引
        lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=5)
        corpus_lsi = lsi[corpus_tfidf] 
        # LSI潛在語義索引
        lsi.save('Aidata/CorpousData_content_cut.lsi')
        corpora.MmCorpus.serialize('Aidata/CorpousData_content_cut.mm', corpus_lsi)
        print("LSI topics:")
        print(lsi.print_topics(5))


    def SearchSimilar(self,doc):
        #content=input("輸入查詢字:")
        content=doc
        # 將標點符號去掉
        punct = set(u''':!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻︽︿﹁﹃﹙﹛﹝（｛“‘-—_…~/ －＊➜■─★☆=@<>◉é''')
        # 去掉網址 ptt的文章內容基本上都會換行 我們順便把最後的換行字元去掉
        content = re.sub(r'https?:\/\/.*[\r\n]*', '', content)
        # 使用 filter 去掉標點符號
        content = " ".join(filter(lambda x: x not in punct, jieba.cut(content)))
        # 去掉換行符號
        doc = content.replace("\n", "").replace("\r", "")
        vec_bow = dictionary.doc2bow(doc.split())
        vec_lsi = lsi[vec_bow]
        print("vec_lsi")
        print(vec_lsi)
        index = similarities.MatrixSimilarity(lsi[corpus],num_features=10)
        index.save("Aidata/CorpousData_content_cut.index") 
        # 計算相似度（前五名）
        sims = index[vec_lsi] 
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        similerDoc=dict(sims[:5]).keys()
        #print(dict(sims[:5]))
        print(similerDoc)
        #for nid in similerDoc:
        #    df=OpenDatadbconnect()
        #    #print(df.iloc[nid])
        #    #print('*'*50)
        #    searchanswer=df.iloc[nid]
        return similerDoc
    
    #OpenDatadbconnect()
    #LoadCorpus()
    #CreateLsi()
    #SearchSimilar()