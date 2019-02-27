import sys
import pandas as pandas
import matplotlib.pyplot as plt
import pylab as pylab
import numpy as numpy
import seaborn as sns 

from tabulate import tabulate
from sklearn.model_selection import train_test_split

#TODO Confusion Matrix -- Distribucion de puntuaciones para el modelo

def main():
    # Read data from file
    df = pandas.read_csv("whitewine.csv", sep=';')
    #explore_data(df)

    y = df[['quality']].copy()
    y = y.applymap(lambda x: 1 if x > 5 else 0)

    df = df.drop(columns=['quality'])
    #print(tabulate(df, headers='keys'))
    print(tabulate(y, headers='keys'))

    #Split into train - test data
    x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.25)


    #Train


    #Accuracy


    #Plot

def explore_data(df):
    # Taken from https://lukesingham.com/whos-going-to-leave-next/
    #Correlation map
    correlation = df.corr()  
    plt.figure(figsize=(10, 10))  
    sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')
    plt.show()

    #Search for missing values
    print(df.apply(lambda x: sum(x.isnull()), axis=0))



if __name__ == '__main__':
    main()