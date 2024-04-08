import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KernelDensity
from sklearn.metrics import accuracy_score
import warnings
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import colors
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import PostureDetecting.CreateDataSet as cds


def WriteScatter(datapd):
    colors = ['red', 'blue']
    labels = [0,1]

    fig, ax = plt.subplots(3, 1, figsize=(8, 24))
    plt.figure(figsize=(10, 10))

    for count in range(3):
        for i, label in enumerate(labels):
            ax[count].scatter(datapd[datapd['LABEL'] == label][f'X{count+1}'].values, datapd[datapd['LABEL'] == label][f'Y{count+1}'].values,
                        color=colors[i], label=label)
        ax[count].set_xlabel('X1')
        ax[count].set_ylabel('Y1')
        ax[count].set_title(f'График распределения классов {count+1}')
        ax[count].legend()
        ax[count].grid()
    plt.show()