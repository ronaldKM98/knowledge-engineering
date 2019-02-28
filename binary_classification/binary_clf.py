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

debug = False

def random_forest(combinations_iterable,x_train,y_train,x_validation,y_validation):
    result_models = []

    for col in combinations_iterable:
        for i in col:
            if debug:
                print(x_df[list(i)])

            features_train = x_train[list(i)].copy()
            features_validation = x_validation[list(i)].copy()

            forest_cl = classify(features_train.values, y_train.values.ravel());
            score = forest_cl.score(features_validation.values, y_validation.values.ravel())

            result_models.append([list(i),score])

    return result_models


def main():
    df = pandas.read_csv("whitewine.csv", sep=';')

    y = df[['quality']].copy()
    x = df.drop(columns=['quality'])

    combinations_iterable = combinations(list(x))

    x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.30)
    x_validation, x_test, y_validation, y_test = train_test_split(x_test, y_test, test_size=0.70)

    print(type(x_validation))

    result_models = random_forest(combinations_iterable,x_train,y_train,x_validation,y_validation)

    for i in result_models:
        print("Score -> ",i[1]," Combinations -> ",i[0])

def main_test():
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
    for i in range(1, 4):
        combinations.append(itertools.combinations(df, i))

    if debug:
        for col in combinations:
            for i in col:
                print(i.items())

    return combinations



if __name__ == '__main__':
    main()
