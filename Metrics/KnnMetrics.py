import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KernelDensity
from sklearn.metrics import accuracy_score,f1_score
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import PostureDetecting.CreateDataSet as cds
import PostureDetecting.Classificators as cl
import seaborn as sns
from xgboost import XGBClassifier
import optuna
from sklearn. metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
def GetBetterKNumber(testdata, testfeat, traindata, trainfeat, validatedata, validatefeat):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    K_range = np.arange(2, 30, 1)
    acc = [[], [], []]

    for k in K_range:
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(traindata, trainfeat)

        res = neigh.predict(traindata)
        acc[0].append(f1_score(trainfeat, res))

        res = neigh.predict(validatedata)
        acc[1].append(f1_score(validatefeat, res))

        res = neigh.predict(testdata)
        acc[2].append(f1_score(testfeat, res))

    ax.plot(K_range, acc[0], color='cornflowerblue', label="Train")
    ax.plot(K_range, acc[1], color='hotpink', label="Validate")
    ax.plot(K_range, acc[2], color='orangered', label="Test")
    ax.legend(loc="upper right")
    ax.set_title(f'Зависимость F-score от числа ближайших соседей')
    ax.set_xlabel(f'K')
    ax.set_ylabel(f'F-score')

    plt.show()

    f_score = max(acc[1])
    max_h = K_range[acc[1].index(f_score)]
    print(' Max acc = ' + str(round(f_score, 3)) + ' MaxK = ' + str(round(max_h, 3)))

    model = KNeighborsClassifier(n_neighbors=max_h)
    return model, f_score


def GetBetterRNumber(testdata, testfeat, traindata, trainfeat, validatedata, validatefeat):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    K_range = np.arange(2, 30, 1)
    acc = [[], [], []]

    for k in K_range:
        ran = RandomForestClassifier(n_estimators=k)
        ran.fit(traindata, trainfeat)

        res = ran.predict(traindata)
        acc[0].append(f1_score(trainfeat, res))

        res = ran.predict(validatedata)
        acc[1].append(f1_score(validatefeat, res))

        res = ran.predict(testdata)
        acc[2].append(f1_score(testfeat, res))

    ax.plot(K_range, acc[0], color='cornflowerblue', label="Train")
    ax.plot(K_range, acc[1], color='hotpink', label="Validate")
    ax.plot(K_range, acc[2], color='orangered', label="Test")
    ax.legend(loc="upper right")
    ax.set_title(f'Зависимость F-score от количества деревьев')
    ax.set_xlabel(f'K')
    ax.set_ylabel(f'F-score')
    plt.show()


    f_score = max(acc[1])
    max_h = K_range[acc[1].index(f_score)]
    print(' Max acc = ' + str(round(f_score, 3)) + ' MaxK = ' + str(round(max_h, 3)))
    model = RandomForestClassifier(n_estimators=max_h)
    return model, f_score

def GetBetterXGBModel(traindata, trainfeat, validatedata, validatefeat):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    def objective(trial):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'num_class': 1,
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'eta': trial.suggest_float('eta', 0.001, 1,log=True),
            'gamma': trial.suggest_float('gamma', 0.01, 1,log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
        }

        model = XGBClassifier(**params)
        model.fit(traindata, trainfeat)
        y_pred = model.predict(validatedata)
        f1 = f1_score(validatefeat, y_pred, average='macro')
        return f1

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    best_params = study.best_params
    print("Best Hyperparameters:", best_params)
    final_model = XGBClassifier(**best_params)
    final_model.fit(traindata, trainfeat)
    y_pred = final_model.predict(validatedata)
    print(len(y_pred))
    f_score = f1_score(validatefeat, y_pred)

    precision, recall, thresholds = precision_recall_curve(validatefeat, y_pred)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, color='purple')

    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')

    # display plot
    plt.show()

    return final_model,f_score

def GetRegrModel(traindata, trainfeat, validatedata, validatefeat):
    f_list = {}
    f_saga_list=list()
    types = {'lbfgs': ['l2', None], 'liblinear': ['l1', 'l2'], 'newton-cg': ['l2', None],
    'newton-cholesky': ['l2', None], 'sag':  ['l2', None], 'saga': ['elasticnet', 'l1', 'l2', None]}
    type = ['l1', 'l2']
    for solver in types:
        if solver == 'saga':
            for i in range(0,11,1):
                model = LogisticRegression(penalty='elasticnet', solver=solver,l1_ratio=i/10)
                model.fit(traindata, trainfeat)
                test_y_pred = model.predict(validatedata)
                f_saga_list.append(f1_score(test_y_pred, validatefeat))
            print(f_saga_list)
            max_value = max(f_saga_list)
            f_list[f'saga-elasticnet{f_saga_list.index(max_value)}']=max_value
        else:
            for penalty in types[solver]:
                    model = LogisticRegression(penalty=penalty, solver=solver)
                    model.fit(traindata, trainfeat)
                    test_y_pred = model.predict(validatedata)
                    f_list[f'{solver}-{penalty}']=f1_score(test_y_pred, validatefeat)
    print(f_list)


    return model,0.96


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
