from sklearn.linear_model import LinearRegression
import sys
import pandas as pandas
import matplotlib.pyplot as plt
import pylab as pylab
import numpy as numpy
import seaborn as sns
import graphviz
import itertools
from sklearn import tree
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


debug = False
MODELS_NUMBER=5

def random_forest(combinations_iterable,x_train,y_train,x_validation,y_validation):
    result_models = []

    for col in combinations_iterable:
        for i in col:
            if debug:
                print(x_df[list(i)])

            features_train = x_train[list(i)].copy()
            features_validation = x_validation[list(i)].copy()

            forest_reg = regression(features_train.values, y_train.values.ravel());

            y_predict = forest_reg.predict(features_validation)

            rmse = mean_squared_error(y_validation, y_predict)

            result_models.append([list(i),rmse])

    return result_models


def main_test():
    print("")
    lista = ['volatile acidity', 'residual sugar', 'density', 'alcohol']
    print(lista)
    # Read data from file
    df = pandas.read_csv("whitewine.csv", sep=';')
    # explore_data(df)
    y = df[['quality']].copy()

    df = df.drop(columns=['quality'])
    df = df[lista].copy()

    # Split into train - test data
    x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.25)

    # Train
    binary_model = regression(x_train.values, y_train.values.ravel())

    # Accuracy
    score = binary_model.score(x_test.values, y_test.values.ravel())
    print('Score linear regression', score)


def main():
    df = pandas.read_csv("whitewine.csv",sep=";")
    
    y = df[['quality']].copy()
    x = df.drop(columns=['quality'])
    
    combinations_iterable = combinations(list(x))
    
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.30)
    x_validation, x_test, y_validation, y_test = train_test_split(x,y,test_size=0.70)

    result_models = random_forest(combinations_iterable,x_train,y_train,x_validation,y_validation)


    menor = float("inf")
    model = None
    index = -1
    best_models = []
    for i in range(5):
        for model_tuple in result_models:
            if model_tuple[1] < menor:
                menor = model_tuple[1] #Score
                model = model_tuple[0] #List of Arguments
                index = model_tuple

        best_models.append([model,menor])
        result_models.remove(index)
        menor = float("inf")

    print(str(best_models))
    #combinations_sorted = sorted(combinations_iterable,
    #   key=lambda score:combinations_iterable[1])

    for i in range(len(best_models)):
        print("*************************************")
        x_features = best_models[i][0]
        print(str(x_features))

        y_test_total = pandas.concat([y_test, y_validation])
        x_test_total = pandas.concat([x_test, x_validation])
        print(y_test_total.shape)

        features_train = x_train[x_features].copy()
        features_test = x_test_total[x_features].copy()

        model_i = regression(features_train.values, y_train.values.ravel())

        y_predict = model_i.predict(features_test)

        rmse = mean_squared_error(y_test_total, y_predict)

        score = model_i.score(features_test.values, y_test_total.values.ravel())
        print("Model -> " + str(x_features) + " New RMES -> " + str(rmse) + " Old RMSE -> " +
                str(best_models[i][1]))
        print("Score -> "+ str(score))


def main_test():
    print("")
    lista = ['fixed acidity', 'volatile acidity', 'pH', 'sulphates', 'alcohol']
    print(lista)
    # Read data from file
    df = pandas.read_csv("whitewine.csv", sep=';')
    # explore_data(df)
    y = df[['quality']].copy()

    df = df.drop(columns=['quality'])
    df = df[lista].copy()

    x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.25)

    model_i = regression(x_train.values, y_train.values.ravel())
    y_predict = model_i.predict(x_train)
    rmse = mean_squared_error(y_train, y_predict)
    print('Score linear regression', rmse)


def regression(x,y):
    model = LinearRegression().fit(x, y)
    return model



def combinations(df):
    print(len(df))
    #combinations = itertools.combinations(df, 10)
    combinations = []
    for i in range(1, 12):
        combinations.append(itertools.combinations(df, i))

    if debug:
        for col in combinations:
            for i in col:
                print(i.items())

    return combinations

main_test()