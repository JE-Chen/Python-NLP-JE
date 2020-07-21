from NLPCore import NLPCore
a=NLPCore()
a.NLP_StopWords.Read_StopWords()
a.NLP_Model.Load_Model('test.model')
res = a.NLP_Model.Get_Model()
y1 = a.NLP_Model.Model.wv.similarity(u"程式", u"Program")
print(y1)
