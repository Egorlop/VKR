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

def GetBetterKNumber(testdata,testfeat,traindata,trainfeat,validatedata,validatefeat):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    K_range = np.arange(3, 30, 1)
    acc = [[],[],[]]

    for k in K_range:
      neigh = KNeighborsClassifier(n_neighbors=k)
      neigh.fit(traindata,trainfeat)

      res=neigh.predict(traindata)
      acc[0].append(accuracy_score(trainfeat, res))

      res=neigh.predict(validatedata)
      acc[1].append(accuracy_score(validatefeat, res))

      res=neigh.predict(testdata)
      acc[2].append(accuracy_score(testfeat, res))

    ax.plot(K_range, acc[0], color='cornflowerblue', label="Train")
    ax.plot(K_range, acc[1], color='hotpink', label="Validate")
    ax.plot(K_range,acc[2], color = 'orangered',label="Test")
    ax.legend(loc="upper right")
    ax.set_title(f'Зависимость Accuracy от числа ближайших соседей')
    ax.set_xlabel(f'K')
    ax.set_ylabel(f'Accuracy')

    plt.show()

    max_acc = max(acc[1])
    max_h = K_range[acc[1].index(max_acc)]
    print(' Max acc = '+str(round(max_acc,3))+' MaxK = ' +str(round(max_h,3)))

    return max_h

def GetBetterRNumber(testdata,testfeat,traindata,trainfeat,validatedata,validatefeat):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    K_range = np.arange(2, 30, 1)
    acc = [[],[],[]]

    for k in K_range:
      ran = RandomForestClassifier(n_estimators=k)
      ran.fit(traindata,trainfeat)

      res=ran.predict(traindata)
      acc[0].append(accuracy_score(trainfeat, res))

      res=ran.predict(validatedata)
      acc[1].append(accuracy_score(validatefeat, res))

      res=ran.predict(testdata)
      acc[2].append(accuracy_score(testfeat, res))

    ax.plot(K_range, acc[0], color='cornflowerblue', label="Train")
    ax.plot(K_range, acc[1], color='hotpink', label="Validate")
    ax.plot(K_range,acc[2], color = 'orangered',label="Test")
    ax.legend(loc="upper right")
    ax.set_title(f'Зависимость Accuracy от количества деревьев')
    ax.set_xlabel(f'K')
    ax.set_ylabel(f'Accuracy')

    plt.show()

    max_acc = max(acc[1])
    max_h = K_range[acc[1].index(max_acc)]
    print(' Max acc = '+str(round(max_acc,3))+' MaxK = ' +str(round(max_h,3)))

    return max_h

def KnnMeshgrid():
    trainxl, traindatapd, trainfeat = cds.ReadFromExcel('Train')

    bestKNumber = GetBetterKNumber()
    colors = ['peru', 'plum']
    labels = [0,1]
    for k in range(3):
        neigh = cl.KNeighbors(trainxl, trainfeat, bestKNumber)
        neigh.fit([t[3*k:3*k+2] for t in trainxl], trainfeat)
        x_min, x_max = traindatapd[f"X{k+1}"].min(), traindatapd[f"X{k+1}"].max()
        y_min, y_max = traindatapd[f"Y{k+1}"].min(), traindatapd[f"Y{k+1}"].max()
        print(bestKNumber)
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))  # cетка 100х100
        xy = np.vstack([xx.ravel(), yy.ravel()]).T

        Z = neigh.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(10, 6))
        plt.pcolormesh(xx, yy, Z, cmap='GnBu')
        colorbar = plt.colorbar()
        colorbar.set_label('Класс')
        plt.xlabel(f"X{k+1}")
        plt.ylabel(f"Y{k+1}")
        plt.title(f'Формируемые области классов и границы между классами для KNN (K={bestKNumber})')  # Заголовок графика
        for i, label in enumerate(labels):
            plt.scatter(traindatapd[traindatapd['LABEL'] == label][f'X{k+1}'].values, traindatapd[traindatapd['LABEL'] == label][f'Y{k+1}'].values,
                        color=colors[i], label=label)
        plt.show()  # Показать график

def KnnContourf():

    # Создание сетки точек 200x200 для оценки плотности
    x_min, x_max, y_min, y_max = data["x1"].min(), data["x1"].max(), data["x2"].min(), data["x2"].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    xy = np.vstack([xx.ravel(), yy.ravel()]).T

    X = data[['x1', 'x2']].values
    Y = data['label'].values

    h = 1  # Ширина окна Сильвермана

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1,
                                                        random_state=42)  # Разделение на обучающую, валидационную и тестовую выборки
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

    X1 = data[data["label"] == 1][["x1", "x2"]]  # Разделение датасета в зависимости от класса
    X2 = data[data["label"] == 2][["x1", "x2"]]
    X3 = data[data["label"] == 3][["x1", "x2"]]

    for kernel in kernels:
        kde1 = KernelDensity(kernel=kernel, bandwidth=h).fit(
            X1)  # создание kde и обучение для каждого конкретного класса
        kde2 = KernelDensity(kernel=kernel, bandwidth=h).fit(X2)
        kde3 = KernelDensity(kernel=kernel, bandwidth=h).fit(X3)

        scores1 = np.exp(kde1.score_samples(xy).reshape(xx.shape))  #
        scores2 = np.exp(kde2.score_samples(xy).reshape(xx.shape))
        scores3 = np.exp(kde3.score_samples(xy).reshape(xx.shape))

        plt.figure(figsize=(12, 6))
        plt.contourf(xx, yy, scores1, cmap='hot', alpha=0.5)
        plt.colorbar()
        plt.contourf(xx, yy, scores3, cmap='GnBu', alpha=0.5)
        plt.colorbar()
        plt.contourf(xx, yy, scores2, cmap='Greens', alpha=0.5)
        plt.colorbar()
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.title(f"Восстановленные двумерные плотности для \n каждого класса для {kernel} окна при P={round(h, 3)}")

        plt.show()