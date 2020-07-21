from NLP_Main import NLP_Main
from NLP_Model import NLP_Model
from NLP_StopWords import NLP_StopWords


class NLPCore():

    def __init__(self):
        try:
            self.NLP_Main=NLP_Main()
            self.NLP_Model=NLP_Model()
            self.NLP_StopWords=NLP_StopWords()
            self.NLP_RuleBase=NLP_RuleBase()
            self.NLP_Client_Analysis=NLP_Client_Analysis()
        except Exception as Err:
            raise Err
        print('NLPCore Ready')
