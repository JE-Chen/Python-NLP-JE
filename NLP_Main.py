# -*- coding: utf-8 -*-
import os
import jieba
from jieba import analyse
import logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from ckiptagger import WS, POS, NER
from hanziconv import HanziConv


class NLP_Main():

    def __init__(self):

        self.ws = WS("./data", disable_cuda=False)
        self.pos = POS("./data", disable_cuda=False)
        self.ner = NER("./data", disable_cuda=False)

    #斷詞(WS)、詞性標記（POS）、命名實體識別（NER）。
# ---------------------------------------------------------------------------------
    #斷詞(WS)
    def NLP_WS(self,text):
        return self.ws([text])

    # 斷詞(WS)並儲存
    def Ws_Save(self):

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        output = open('wiki_ws.txt', 'w', encoding='utf-8')
        with open('wiki_seg.txt', 'r', encoding='utf-8') as content:
            for texts_num, line in enumerate(content):
                line = line.strip('\n')
                wordss = self.ws([line])
                for words in wordss:
                    for word in words:
                        if(word!=' '):
                            output.write(word+'\t')
                            print(word)
                        else:
                            output.write('\n')

                if (texts_num + 1) % 10000 == 0:
                    logging.info("已完成前 %d 行的斷詞" % (texts_num + 1))
        output.close()
# ---------------------------------------------------------------------------------

    #詞性標記（POS）
    def NLP_POS(self,text):
        return self.pos(self.NLP_WS(text))

    #命名實體識別（NER）
    def NLP_NER(self,text):
        return self.ner(self.NLP_WS(text), self.NLP_POS(text))

# ---------------------------------------------------------------------------------
    '''TF-IDF 提取法
    
    是一種常用於資訊檢索的加權技術，一種統計方法。用於評估在一個文件集中一個詞對某份文件的重要性，一個詞對文件越重要越有可能成為關鍵詞。
    
    TF-IDF演算法由兩部分組成：TF演算法以及IDF演算法。
    
    TF演算法是統計一個詞在一篇文件中出現的頻率(詞頻)。
    
    IDF演算法則是統計一個詞在文件集的多少個文件中出現，即是如果一個詞在越少的文件中出現，則其對文件的區分能力也越強。
    
    '''
    def Extract_Tag_TF_IDF(self,text):
        Array =[]
        for x,w in jieba.analyse.extract_tags(text,withWeight=True):
            Array.append((str(x)+': '+str(w)))
        return Array

    '''TextRank 演算法
    
    TextRank 的前身為Google所開發的PageRank
    PageRank的主要功用是用於衡量網站之間的重要性，透過網頁之間的連結以及各個網頁的投票計算出其重要性。
    TextRank則是透過文章中去尋找其中重要的詞或句子。
    '''
    def Extract_Tag_TextRank(self, text):
        Array = []
        for x,w in jieba.analyse.textrank(text,withWeight=True):
            Array.append((str(x)+': '+str(w)))
        return Array

    '''權重值
    
    權重是一個相對的概念，是針對某一指標而言。
    某一指標的權重是指該指標在整體評價中的相對重要程度。
    
    打個比方說, 一件事情你給它打100分,你的老闆給它打60分, 如果平均則是(100+60)/2=80分。
    但因為老闆說的話分量比你重, 假如老闆的權重是2, 你是1, 這時求平均值就是加權平均了, 結果是(100*1 +60*2)/(1+2)=73.3分 
    '''


#---------------------------------------------------------------------------------
    #轉換簡體至繁體並存檔
    def Transform_ZhTw_Save(self,File_Name,Next_FileName):
        FileRead=[]
        with open(File_Name,'rb') as RawFile:
            for line in RawFile:
                FileRead.append(HanziConv.toTraditional(line))
        with open(Next_FileName,'wb') as Next_File:
            for i in range(len(FileRead)):
                for j in range(len(FileRead[i])):
                    Next_File.write(FileRead[i][j].encode('utf-8'))

    #轉換簡體至繁體
    def Transform_ZhTw(self,Text):
        return HanziConv.toTraditional(Text)

    #轉換繁體至簡體
    def Transform_Ch(self,Text):
        return HanziConv.toSimplified(Text)


