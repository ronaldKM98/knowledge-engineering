from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import sys
import pandas as pandas
import matplotlib.pyplot as plt
import pylab as pylab
import numpy as numpy
import seaborn as sns
import graphviz
import itertools
import sklearn
from sklearn import linear_model
from sklearn import tree
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
# TODO Confusion Matrix -- Distribución de puntuaciones para el modelo

debug = False

def main(path):
    # Read data from file
    test = pandas.read_csv(path, sep=';')
    df = pandas.read_csv("whitewine.csv", sep=';')
    # explore_data(df)

    y = df[['quality']].copy()
    y1 = y
    y = y.applymap(lambda x: 1 if x > 5 else 0)
    df = df.drop(columns=['quality'])
    
    logistic_regression_model = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',  'density', 'pH', 'sulphates', 'alcohol']
    tree_model = ['fixed acidity',  'volatile acidity', 'citric acid', 'residual sugar',  'pH', 'sulphates',  'alcohol']
    lineal_regression = ['fixed acidity',  'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'pH', 'sulphates', 'alcohol']
    #print(tabulate(df, headers='keys'))
    #print(tabulate(y, headers='keys'))
    df_logistic = df[logistic_regression_model].copy()
    df_tree = df[tree_model].copy()
    df_lineal = df[lineal_regression].copy()

    # Train
    tree_clf = tree.DecisionTreeClassifier()
    clf = linear_model.LogisticRegression(
        solver='lbfgs', multi_class='multinomial', max_iter=3000, C=10)
    lineal_reg = linear_model.LinearRegression()
    
    scores = cross_val_score(clf, df.values, y.values.ravel(), cv=10)
    scores_tree = cross_val_score(tree_clf, df.values, y.values.ravel(), cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print("Accuracy Tree: %0.2f (+/- %0.2f)" % (scores_tree.mean(), scores_tree.std() * 2))

    # Train
    tree_clf = tree_clf.fit(df_tree.values, y.values.ravel())
    clf = clf.fit(df_logistic.values, y.values.ravel())
    reg = lineal_reg.fit(df_lineal.values, y1.values.ravel())
    

    # Predict desicion tree
    predicted = pandas.DataFrame(tree_clf.predict(test[tree_model].copy().values))
    predicted = predicted.applymap(lambda x: "MALO" if x == 0 else "BUENO")
    # predict lineal
    y_predicted = pandas.DataFrame(reg.predict(test[lineal_regression].copy().values))
    # print
    frames = [predicted, y_predicted]
    result = pandas.concat(frames, axis=1, sort=False)
    result.to_csv("output.csv", index=False, sep=',')

    
    

if __name__ == '__main__':
    main(sys.argv[1])