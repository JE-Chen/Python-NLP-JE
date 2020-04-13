from sklearn import preprocessing

enc=preprocessing.OneHotEncoder()
enc.fit([
    [0,0,3],
    [1,1,0],
    [0,2,1],
    [1,0,2]])
array=enc.transform([[0,1,3]]).toarray()
print(array)

'''10 010  0001

陣列必須直著看 有幾個不同元素代表有幾維 0 1 為 2　，　0 1 2 為 3 　
1開始 有幾維後面接幾個0 ex 0 1 = 10 01 ， 0 1 2 = 100 010 001
10 01 [0,1]
100 010 001 [0,1,2] 
1000 0100 0010 0001 [0,1,2,3]

'''