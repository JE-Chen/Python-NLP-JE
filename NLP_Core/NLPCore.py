import datetime

from Models.NLP_Main import NLP_Main
from Models.NLP_Model import NLP_Model
from Models.NLP_StopWords import NLP_StopWords

class NLPCore():

    def __init__(self):
        try:
            self.NLP_Main=NLP_Main()
            self.NLP_Model=NLP_Model()
            self.NLP_StopWords=NLP_StopWords()
        except Exception as Err:
            raise Err
        print(datetime.datetime.now(),self.__class__,'Ready',sep=' ')
