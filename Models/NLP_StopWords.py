import jieba
import warnings
import os
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# from ckiptagger import WS, POS, NER
class NLP_StopWords():

    def __init__(self):

        '''
        self.ws = WS("./data", disable_cuda=False)
        self.pos = POS("./data", disable_cuda=False)
        self.ner = NER("./data", disable_cuda=False)
        '''
        self.StopWords=[]
        self.Segments=[]
        self.RemainderWords=[]

# ---------------------------------------------------------------------------------
    def Read_StopWords(self):
        with open('StopWord.txt', 'r', encoding='UTF-8') as file:
            for data in file.readlines():
                data = data.strip()
                self.StopWords.append(data)

# ---------------------------------------------------------------------------------
    def Get_After_StopWords_File(self):
        with open('text.txt', 'r', encoding='UTF-8') as file:
            #讀入文檔
            text = file.read()

            #結巴中文斷詞
            self.Segments = jieba.cut(text, cut_all=False)
        self.RemainderWords = list(filter(lambda a: a not in self.StopWords and a != '\n',  self.Segments))
        return self.RemainderWords

# ---------------------------------------------------------------------------------
    def Get_After_StopWords_Text(self,text):

        #結巴中文斷詞
        self.Segments = jieba.cut(text, cut_all=False)

        #self.Segments = self.ws([text])
        print(self.Segments)


        self.RemainderWords = list(filter(lambda a: a not in self.StopWords and a != '\n',  self.Segments))
        return self.RemainderWords