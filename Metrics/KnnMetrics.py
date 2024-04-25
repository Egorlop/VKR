import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KernelDensity
from sklearn.metrics import accuracy_score
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import PostureDetecting.CreateDataSet as cds
import PostureDetecting.Classificators as cl


def GetBetterKNumber(testdata, testfeat, traindata, trainfeat, validatedata, validatefeat):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    K_range = np.arange(3, 10, 1)
    acc = [[], [], []]

    for k in K_range:
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(traindata, trainfeat)

        res = neigh.predict(traindata)
        acc[0].append(accuracy_score(trainfeat, res))

        res = neigh.predict(validatedata)
        acc[1].append(accuracy_score(validatefeat, res))

        res = neigh.predict(testdata)
        acc[2].append(accuracy_score(testfeat, res))

    ax.plot(K_range, acc[0], color='cornflowerblue', label="Train")
    ax.plot(K_range, acc[1], color='hotpink', label="Validate")
    ax.plot(K_range, acc[2], color='orangered', label="Test")
    ax.legend(loc="upper right")
    ax.set_title(f'Зависимость Accuracy от числа ближайших соседей')
    ax.set_xlabel(f'K')
    ax.set_ylabel(f'Accuracy')

    plt.show()

    max_acc = max(acc[1])
    max_h = K_range[acc[1].index(max_acc)]
    print(' Max acc = ' + str(round(max_acc, 3)) + ' MaxK = ' + str(round(max_h, 3)))

    return max_h, max_acc


def GetBetterRNumber(testdata, testfeat, traindata, trainfeat, validatedata, validatefeat):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    K_range = np.arange(3, 10, 1)
    acc = [[], [], []]

    for k in K_range:
        ran = RandomForestClassifier(n_estimators=k)
        ran.fit(traindata, trainfeat)

        res = ran.predict(traindata)
        acc[0].append(accuracy_score(trainfeat, res))

        res = ran.predict(validatedata)
        acc[1].append(accuracy_score(validatefeat, res))

        res = ran.predict(testdata)
        acc[2].append(accuracy_score(testfeat, res))

    ax.plot(K_range, acc[0], color='cornflowerblue', label="Train")
    ax.plot(K_range, acc[1], color='hotpink', label="Validate")
    ax.plot(K_range, acc[2], color='orangered', label="Test")
    ax.legend(loc="upper right")
    ax.set_title(f'Зависимость Accuracy от количества деревьев')
    ax.set_xlabel(f'K')
    ax.set_ylabel(f'Accuracy')

    plt.show()

    max_acc = max(acc[1])
    max_h = K_range[acc[1].index(max_acc)]
    print(' Max acc = ' + str(round(max_acc, 3)) + ' MaxK = ' + str(round(max_h, 3)))

    return max_h, max_acc


def KnnMeshgrid():
    trainxl, traindatapd, trainfeat = cds.ReadFromExcel('Train')

    bestKNumber = GetBetterKNumber()
    colors = ['peru', 'plum']
    labels = [0, 1]
    for k in range(3):
        neigh = cl.KNeighbors(trainxl, trainfeat, bestKNumber)
        neigh.fit([t[3 * k:3 * k + 2] for t in trainxl], trainfeat)
        x_min, x_max = traindatapd[f"X{k + 1}"].min(), traindatapd[f"X{k + 1}"].max()
        y_min, y_max = traindatapd[f"Y{k + 1}"].min(), traindatapd[f"Y{k + 1}"].max()
        print(bestKNumber)
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))  # cетка 100х100
        xy = np.vstack([xx.ravel(), yy.ravel()]).T

        Z = neigh.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(10, 6))
        plt.pcolormesh(xx, yy, Z, cmap='GnBu')
        colorbar = plt.colorbar()
        colorbar.set_label('Класс')
        plt.xlabel(f"X{k + 1}")
        plt.ylabel(f"Y{k + 1}")
        plt.title(
            f'Формируемые области классов и границы между классами для KNN (K={bestKNumber})')  # Заголовок графика
        for i, label in enumerate(labels):
            plt.scatter(traindatapd[traindatapd['LABEL'] == label][f'X{k + 1}'].values,
                        traindatapd[traindatapd['LABEL'] == label][f'Y{k + 1}'].values,
                        color=colors[i], label=label)
        plt.show()  # Показать график
