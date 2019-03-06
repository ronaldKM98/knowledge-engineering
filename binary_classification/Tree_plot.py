import graphviz
from sklearn import tree
import pandas as pandas

params_list = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'free sulfur dioxide', 'sulphates', 'alcohol']
df = pandas.read_csv("whitewine.csv",sep=';')

y = df[['quality']].copy()
x = df.drop(columns=['quality'])

x = df[params_list].copy()

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x.values,y.values.ravel())


dot_data = tree.export_graphviz(clf, out_file=None, 
                feature_names=list(x),  
                class_names='quality',  
                filled=True, rounded=True,  
                special_characters=True)  

graph = graphviz.Source(dot_data, format='svg')  
graph.render()