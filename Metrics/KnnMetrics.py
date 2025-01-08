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
from os import listdir
from os.path import isfile, join

def PreasureGist():
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.set_title(f'Давление на межпозвоночные диски по сравнению с положением стоя')
    ax.set_xlabel(f'Положение')
    ax.set_ylabel(f'Давление')
    testacc = [25, 100,140,185]
    info = ['лёжа на спине', 'стоя', 'сидя', 'сидя, с наклоном вперед']
    plt.bar(info, testacc, width=0.8, color='orange')
    for x, y in zip(info, testacc):
        plt.text(x, (y / 2)-4, str(y)+'%', ha='center', va='bottom')
    plt.show()
def TestClassificator(classificator):
    testres = []
    testacc = []
    for k in range(1,6):
        testfiles = [f for f in listdir(f'D:\\pythonProject\\datasets\\TestDatasets\\Test{k}') if
                     isfile(join(f'D:\\pythonProject\\datasets\\TestDatasets\\Test{k}', f))]
        for i in range(len(testfiles)):
            if i == 0:
                data = pd.read_excel(f'D:\\pythonProject\\datasets\\TestDatasets\\Test{k}\\{testfiles[i]}')
            else:
                data = pd.concat([data, pd.read_excel(f'D:\\pythonProject\\datasets\\TestDatasets\\Test{k}\\{testfiles[i]}')])
        testfromxl = []
        testfeat = data['LABEL'].values
        for i in range(1, 4):
            for sym in ['X', 'Y', 'Z']:
                testfromxl.append(data[sym + str(i)].values)
        testdata = np.transpose(testfromxl)
        res = classificator.predict(testdata)
        testres.append(round(f1_score(testfeat, res),4))
        testacc.append(round(accuracy_score(testfeat, res), 4))
    print(testres)
    print(testacc)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.set_title(f'Точность приложения на тестовых выборках')
    ax.set_xlabel(f'Number of Dataset')
    ax.set_ylabel(f'Accuracy')
    info = ['TestDataset 1', 'TestDataset 2','TestDataset 3','TestDataset 4','TestDataset 5']
    plt.bar(info, testacc, width=0.8, color='orange')
    for x, y in zip(info, testacc):
        plt.text(x, (y/2)+0.03, y, ha='center', va='bottom')
    plt.show()
def GetBetterKNumber(testdata, testfeat, traindata, trainfeat, validatedata, validatefeat):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    K_range = np.arange(2, 10, 1)
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
    ax.set_title(f'Зависимость F-score от числа ближайших соседей при использовании координат X-Y-Z')
    ax.set_xlabel(f'K')
    ax.set_ylabel(f'F-score')

    plt.show()

    f_score = max(acc[1])
    max_h = K_range[acc[1].index(f_score)]
    print(' Max acc = ' + str(round(f_score, 3)) + ' MaxK = ' + str(round(max_h, 3)))

    model = KNeighborsClassifier(n_neighbors=max_h)
    return model, f_score,'K = ' + str(max_h)


def GetBetterRNumber(testdata, testfeat, traindata, trainfeat, validatedata, validatefeat):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    K_range = np.arange(2, 10, 1)
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
    ax.set_title(f'Зависимость F-score от количества деревьев при использовании координат X-Y-Z')
    ax.set_xlabel(f'K')
    ax.set_ylabel(f'F-score')
    plt.show()


    f_score = max(acc[1])
    max_h = K_range[acc[1].index(f_score)]
    print(' Max acc = ' + str(round(f_score, 3)) + ' MaxK = ' + str(round(max_h, 3)))
    model = RandomForestClassifier(n_estimators=max_h)

    return model, f_score, 'R = ' + str(max_h)

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
    return final_model,f_score

def GetRegrModel(traindata, trainfeat, validatedata, validatefeat):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    l1_rates = []
    f_list = {}
    f2_list = []
    f_saga_list=list()
    types = {'lbfgs': ['l2', None], 'liblinear': ['l1', 'l2'], 'newton-cg': ['l2', None],
            'newton-cholesky': ['l2', None], 'sag':  ['l2', None],
            'saga': ['elasticnet', 'l1', 'l2', None]}
    for solver in types:
        for_solver = []
        if solver == 'saga':
            for i in range(0,11,1):
                l1_rates.append(i/10)
                model = LogisticRegression(penalty='elasticnet', solver=solver,l1_ratio=i/10)
                model.fit(traindata, trainfeat)
                test_y_pred = model.predict(validatedata)
                f_saga_list.append(f1_score(test_y_pred, validatefeat))
            max_value = [max(f_saga_list)]
            f_list[f'saga-{f_saga_list.index(max_value[0])}']=[round(max(f_saga_list),7)]
            l1 = f_saga_list.index(max_value[0])
            f2_list.append(round(max(f_saga_list),7))
        else:
            for penalty in types[solver]:
                model = LogisticRegression(penalty=penalty, solver=solver)
                model.fit(traindata, trainfeat)
                test_y_pred = model.predict(validatedata)
                for_solver.append(round(f1_score(test_y_pred, validatefeat),7))
                f2_list.append(round(f1_score(test_y_pred, validatefeat),7))
            f_list[f'{solver}']=for_solver
    ax.plot(l1_rates, f_saga_list, color='cornflowerblue', label="Validation",linewidth=2)
    ax.legend(loc="upper left")
    ax.set_title(f'Зависимость F-score от L1-ratio для LogisticRegression(ElasticNet) при использовании координат X-Y-Z')
    ax.set_xlabel(f'L1-ratio')
    ax.set_ylabel(f'F-score')
    plt.show()

    ax = plt.subplot(111)
    solvers=f_list.keys()
    ind = 1
    for solver in solvers:
        if  'saga' in solver:
            ax.bar(ind , f_list[solver], width=0.6, color='blue', align='center')
        else:
            ax.bar(ind-0.2, f_list[solver][0], width=0.4, color='orange', align='center')
            ax.bar(ind+0.2, f_list[solver][1], width=0.4, color='blue', align='center')
        ind+=1

    plt.xticks([1, 2, 3,4,5,6], ['lbfgs', 'liblinear', 'newton-cg', 'newton-ch', 'sag', 'saga'])
    ax.set_title(f'Зависимость F-score от разных \nконфигураций LogisticRegression при использовании координат X-Y-Z')
    ax.set_xlabel(f'Solver')
    ax.set_ylabel(f'F-score')

    max_value = max(f2_list)
    res=1
    for solver in f_list:
        ind=0
        for score in f_list[solver]:
            if solver=='saga-10':
                plt.text(res, score, 'elastic', ha='center', va='bottom')
            else:
                if types[solver][ind] == None:
                    label='None'
                else:
                    label = types[solver][ind]
                plt.text(res -0.2 + ind*0.4, score,label, ha='center', va='bottom')
            if max_value==score:
                if solver == 'saga':
                    model = LogisticRegression(penalty='elasticnet', solver=solver, l1_ratio=l1)
                    model.fit(traindata, trainfeat)
                    test_y_pred = model.predict(validatedata)
                else:
                    model = LogisticRegression(penalty=types[solver][ind], solver=solver)
                    model.fit(traindata, trainfeat)
                    test_y_pred = model.predict(validatedata)
            ind+=1
        res+=1

    plt.show()
    print(model)
    print(f2_list)
    print(max_value)
    return model,max_value


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
            f'Формируемые области классов и границы между классами для KNN (K={bestKNumber}) при использовании координат X-Y-Z')  # Заголовок графика
        for i, label in enumerate(labels):
            plt.scatter(traindatapd[traindatapd['LABEL'] == label][f'X{k + 1}'].values,
                        traindatapd[traindatapd['LABEL'] == label][f'Y{k + 1}'].values,
                        color=colors[i], label=label)
        plt.show()  # Показать график
