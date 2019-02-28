import sys
import pandas as pandas
import matplotlib.pyplot as plt
import pylab as pylab
import numpy as numpy
import seaborn as sns
import graphviz
import itertools

from sklearn import linear_model
from sklearn import tree
from tabulate import tabulate
from sklearn.model_selection import train_test_split

# TODO Confusion Matrix -- Distribucion de puntuaciones para el modelo


def main():
    # Read data from file
    df = pandas.read_csv("whitewine.csv", sep=';')
    # explore_data(df)
    
    y = df[['quality']].copy()
    y = y.applymap(lambda x: 1 if x > 5 else 0)

    df = df.drop(columns=['quality'])
    combinations(list(df))
    #print(tabulate(df, headers='keys'))
    #print(tabulate(y, headers='keys'))

    # Split into train - test data
    x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.25)

    # Train
    binary_model = classify(x_train.values, y_train.values.ravel())
    tree_model = tree_classifier(x_train.values, y_train.values.ravel())

    # Accuracy
    score = binary_model.score(x_test.values, y_test.values.ravel())
    tree_score = tree_model.score(x_test.values, y_test.values.ravel())
    print('Score logistic regression', score)
    print('Score decision tree', tree_score)

    # Plot


def explore_data(df):
    # Taken from https://lukesingham.com/whos-going-to-leave-next/
    # Correlation map
    correlation = df.corr()
    plt.figure(figsize=(10, 10))
    sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')
    plt.show()

    # Good vs bad wines
    y = df[['quality']].copy()
    y = y.applymap(lambda x: 1 if x > 5 else 0)  # Map function over dataframe
    print('Occurrences of each class. 1 for good wines')
    print(y['quality'].value_counts(), '\n')

    # Search for missing values
    print('Search for missing values')
    print(df.apply(lambda x: sum(x.isnull()), axis=0), '\n')


def classify(features_train, labels_train):
    clf = linear_model.LogisticRegression(
        solver='lbfgs', multi_class='multinomial', max_iter=1500)
    clf.fit(features_train, labels_train)
    return clf


def tree_classifier(features_train, labels_train):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(features_train, labels_train)

    '''dot_data = tree.export_graphviz(clf, out_file=None, feature_names=iris.feature_names,
                                    class_names=iris.target_names,
                                    filled=True, rounded=True,
                                    special_characters=True)


    graph = graphviz.Source(dot_data)
    graph.render("iris")
    graph'''
    return clf


def combinations(df):
    print(len(df))
    #combinations = itertools.combinations(df, 10)
    combinations = []
    for i in range(1, 11):
        combinations.append(itertools.combinations(df, i))
    

    for col in combinations:
        for i in col:
            print(i)
    
    
    

if __name__ == '__main__':
    main()
