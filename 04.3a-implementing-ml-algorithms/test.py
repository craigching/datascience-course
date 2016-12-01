import pandas as pd
from collections import Counter
#from LogisticRegression2 import LogisticRegression
#from LogisticRegression import LogisticRegression
from MyLogisticRegression import MyLogisticRegression

wine = pd.read_csv('wine.csv')
y = (wine.Type=='Red').values.astype(int)
X = wine.loc[:,wine.columns[0:12]].values


#t = pd.Series([tuple(i) for i in wine.Type])
#counts = t.value_counts()
#print(counts)

#print(y)
#print('===')
#print(y.reshape(y.size,1))
#print('===')
#print(X)


#l = LogisticRegression(X,y,tolerance=1e-6)
#l.gradient_decent(alpha=1e-2,max_iterations=1e4)
l = MyLogisticRegression(X, y, epsilon = 1e-6)
l.optimize(alpha = 1e-2, max_iterations = 1e4)
