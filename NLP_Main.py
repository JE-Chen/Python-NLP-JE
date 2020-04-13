# -*- coding: utf-8 -*-
import os
import jieba
from jieba import analyse
import warnings
import logging
from gensim.corpora import WikiCorpus
from gensim.models import word2vec
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')
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
        text = text
        ws_results = self.ws([text])
        return ws_results

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
        pos_results = self.pos(self.NLP_WS(text))
        return pos_results

    #命名實體識別（NER）
    def NLP_NER(self,text):
        ner_results = self.ner(self.NLP_WS(text), self.NLP_POS(text))
        return ner_results

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

#---------------------------------------------------------------------------------
    #從下載下來的維基提取資料 並存到Save_File
    def Corpus_Wiki(self,WikiCorpus_File_Name='zhwiki-20200301-pages-articles.xml.bz2',Save_File="wiki_texts.txt"):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        wiki_corpus = WikiCorpus(WikiCorpus_File_Name, dictionary={})
        texts_num = 0

        with open(Save_File, 'w', encoding='utf-8') as output:
            for text in wiki_corpus.get_texts():
                output.write(' '.join(text) + '\n')
                texts_num += 1
                if texts_num % 10000 == 0:
                    logging.info("已處理 %d 篇文章" % texts_num)
# ---------------------------------------------------------------------------------
    def Train_Model(self,Model_Name="word2vec.model",*args):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        sentences = word2vec.LineSentence("wiki_seg.txt")
        '''
        window:還記得孔乙己的例子嗎？能往左往右看幾個字的意思
        sentences:當然了，這是要訓練的句子集，沒有他就不用跑了
        size：特徵向量的維度，通常設為300 
        sg:這個不是三言兩語能說完的，sg=1表示採用skip-gram,sg=0 表示採用cbow
        alpha:機器學習中的學習率，這東西會逐漸收斂到 min_alpha
        min_count：字詞出現少於這個閥值(threshold)則捨棄
        max_vocab_size：RAM的限制，如超過上限則捨棄不頻
        繁使用的， None 為不限制
        Sample: 高頻字詞的取樣率
        seed : 亂數產生器，與初始化向量有關係
        Workers: 多執行緒的數量
        iter : 迭代次數
        batch_words : 每個batch的字詞量
        '''
        model = word2vec.Word2Vec(sentences, size=250)
        #保存模型，供日後使用
        model.save(Model_Name)
        #模型讀取方式
        # model = word2vec.Word2Vec.load("your_model_name")