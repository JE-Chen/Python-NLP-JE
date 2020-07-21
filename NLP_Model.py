import warnings
import logging
import datetime
from gensim.corpora import WikiCorpus
from gensim.models import word2vec
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')

class NLP_Model():

    def __init__(self):
        self.Model=None
        self.Load_Model()
        print(datetime.datetime.now())
        print('\n\n')

# ---------------------------------------------------------------------------------
    #讀取模型
    def Load_Model(self,Path='./Model/word2vec.model'):
        self.Model = word2vec.Word2Vec.load(Path)
        return self.Model

    def Get_Model(self):
        return self.Model

# ---------------------------------------------------------------------------------
    #訓練模型
    def Train_Model(self, Model_Name="word2vec.model",size=300,Train_txt="wiki_seg.txt" ,*args):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        sentences = word2vec.LineSentence(Train_txt)
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
        if(len(args)==2):
            self.Model = word2vec.Word2Vec(sentences, size=size,min_count=1)
        else:
            self.Model = word2vec.Word2Vec(sentences, size=size,min_count=1)
        # 保存模型，供日後使用
        self.Model.save(Model_Name)
        # 模型讀取方式
        # model = word2vec.Word2Vec.load("your_model_name")

# ---------------------------------------------------------------------------------
    # 從下載下來的維基提取資料 並存到Save_File
    def Corpus_Wiki(self, WikiCorpus_File_Name='zhwiki-20200301-pages-articles.xml.bz2',
                    Save_File="wiki_texts.txt"):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        wiki_corpus = WikiCorpus(WikiCorpus_File_Name, dictionary={})
        texts_num = 0

        with open(Save_File, 'w', encoding='utf-8') as output:
            for text in wiki_corpus.get_texts():
                output.write(' '.join(text) + '\n')
                texts_num += 1
                if texts_num % 10000 == 0:
                    logging.info("已處理 %d 篇文章" % texts_num)