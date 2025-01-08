from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import QMainWindow
# from UI.posture_detector import Ui_Detector
import PostureDetecting.CreateDataSet as cds
from UI.altui_main import Ui_Detector
import PostureDetecting.Classificators as cl
import Metrics.KnnMetrics as km
import PostureDetecting.CreateDataSet as cds
from sklearn.model_selection import train_test_split
import mediapipe as mp
import cv2
import time
from plyer import notification
import chime
import pandas as pd
import GoogleDrive.UploadAndDownloadDrive as uadd
import os
import serial
import matplotlib.pyplot as plt

# s = serial.Serial('COM11')
start = time.time()
end = time.time()
vibr_start = time.time()
vibr_end = time.time()
pointscoords = []
sound_flg = 0
vibr_flg = 0
mes_flg = 0
custom_flg = 0

def SplitDataset(datasetxl, datasetfeat):
    x, testdata, y, testfeat = train_test_split(datasetxl, datasetfeat, test_size=0.2, train_size=0.8)
    traindata, validatedata, trainfeat, validatefeat = train_test_split(x, y, test_size=0.2, train_size=0.8)
    print(len(testfeat), len(trainfeat),len(validatefeat))
    return testdata, testfeat, traindata, validatedata, trainfeat, validatefeat


def PickClassificator(testdata, testfeat, traindata, trainfeat, validatedata, validatefeat):
    km.PreasureGist()
    bestRegrModel, bestRegrModelscore = km.GetRegrModel(traindata, trainfeat, validatedata, validatefeat)
    bestRandModel, bestRandModelscore, infoR = km.GetBetterRNumber(testdata, testfeat, traindata, trainfeat, validatedata, validatefeat)
    bestKNeiModel, bestKNeiModelscore,infoK = km.GetBetterKNumber(testdata, testfeat, traindata, trainfeat, validatedata, validatefeat)
    bestXGBModel, bestXGBModelscore = km.GetBetterXGBModel(testdata, testfeat, traindata, trainfeat)
    f_measures = {'KNeighbors':round(bestKNeiModelscore,4), 'RandomForest':round(bestRandModelscore,4),
                  'XGBoost':round(bestXGBModelscore,4),'LogisticRegression':round(bestRegrModelscore,4)}
    maximum_value = max(f_measures.values())
    max_key = [k for k,v in f_measures.items() if v == maximum_value][0]
    info = [infoK,infoR,' ',' ']
    print(f_measures, max_key)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.set_title(f'Максимальное значение F-score при использовании координат X-Y-Z')
    ax.set_xlabel(f'Model')
    ax.set_ylabel(f'F-score')
    plt.bar(f_measures.keys(), f_measures.values(), width=0.8, color=['#ffa3ad','#ffe15f','#438add','#ffc25c'])
    for x, y, z in zip(f_measures.keys(), f_measures.values(),info):
        plt.text(x, (y/2)+0.03,y, ha='center', va='bottom')
        plt.text(x, (y/2)-0.02, z, ha='center', va='bottom')
    plt.show()
    if max_key == 'LogisticRegression':
        classificator = bestRegrModel
    elif max_key == 'RandomForest':
        classificator = bestRandModel
    elif max_key == 'KNeighbors':
        classificator = bestKNeiModel
    elif max_key == 'XGBoost':
        classificator = bestXGBModel
    km.TestClassificator(classificator)
    return classificator


class MainWindow(QMainWindow):
    def __init__(self):
        self.classificator = 1
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.timer = QTimer()
        QMainWindow.__init__(self)
        self.ui = Ui_Detector()
        self.ui.setupUi(self)

        self.ui.btn_page_3.clicked.connect(self.prepare_statistic)
        self.ui.btn_page_3.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page_3))
        self.ui.comboBox.currentTextChanged.connect(self.out_statistic)
        self.ui.btn_page_3.clicked.connect(self.finish_detecting)

        self.ui.btn_page_1.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page_1))
        self.ui.btn_start_creating.clicked.connect(self.start_creating)
        self.ui.btn_page_1.clicked.connect(self.finish_detecting)
        self.ui.btn_stop_creating.clicked.connect(self.stop_button)
        self.ui.btn_stop_creating.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page_1))

        self.ui.btn_page_2.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page_2))
        self.ui.btn_start_detecting.clicked.connect(self.start_detecting)
        self.ui.btn_stop_detecting.clicked.connect(self.finish_detecting)

        self.ui.checkBoxSound.stateChanged.connect(self.sound_checkbox_changed)
        self.ui.checkBoxVibr.stateChanged.connect(self.vibr_checkbox_changed)
        self.ui.checkBoxMes.stateChanged.connect(self.mes_checkbox_changed)
        self.ui.checkBoxCust.stateChanged.connect(self.custom_checkbox_changed)

        self.ui.stackedWidget.setCurrentWidget(self.ui.page_4)
        self.ui.btn_page_4.clicked.connect(self.download_datasets)

    def out_statistic(self, period):

        cds.CreateStatsImage(period)
        if period == 'Месяц':
            print(period)
            self.ui.image_label_3.setPixmap(QPixmap(u"Month.png"))
        if period == 'Неделя':
            print(period)
            self.ui.image_label_3.setPixmap(QPixmap(u"Week.png"))
        if period == 'Квартал':
            print(period)
            self.ui.image_label_3.setPixmap(QPixmap(u"Quat.png"))

    def prepare_statistic(self):
        self.ui.comboBox.addItems(['Неделя', 'Месяц', 'Квартал'])
        # self.ui.comboBox.setCurrentText('Месяц')
        cds.CreateStatsImage('Неделя')
        self.ui.image_label_3.setPixmap(QPixmap(u"Week.png"))

    def download_datasets(self):
        uadd.DownloadFromDrive()

    def sound_checkbox_changed(self, state):
        global sound_flg
        if state == 2:
            sound_flg = 1
        else:
            sound_flg = 0
        print('sound')

    def vibr_checkbox_changed(self, state):
        global vibr_flg
        if state == 2:
            vibr_flg = 1
        else:
            vibr_flg = 0
        print('vibr')

    def mes_checkbox_changed(self, state):
        global mes_flg
        if state == 2:
            mes_flg = 1
        else:
            mes_flg = 0
        print('mes')

    def custom_checkbox_changed(self, state):
        global custom_flg
        print(state)
        if state == 2:
            custom_flg = 1
        else:
            custom_flg = 0
        print('cust')

    def finish_detecting(self):
        if self.timer.isActive():
            self.timer.stop()
        self.ui.image_label.setText(QCoreApplication.translate("Detector",
                                                               u"<html><head/><body><p align=\"center\"></p></body></html>",
                                                               None))

    def create_classificator(self, type):
        if type == 'All':
            validatedata,validatefeat,traindata,trainfeat,testdata,testfeat = cds.ReadFromExcel('All')
        else:
            datasetxl, datasetpd, datasetfeat = cds.ReadFromExcel('Custom')
        print(len(validatedata))
        print(len(traindata))
        print(len(testdata))
        self.classificator = PickClassificator(testdata, testfeat, traindata, trainfeat, validatedata, validatefeat)

    def start_detecting(self):
        global start
        global end
        global custom_flg
        print(custom_flg)
        if custom_flg == 1:
            print(custom_flg)
            type = 'Custom'
        else:
            type = 'All'
        self.create_classificator(type)
        start = time.time()
        end = time.time()
        self.timer = QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.display_detecting_video_stream)
        self.cap = cv2.VideoCapture(0)
        self.timer.start()

    def display_detecting_video_stream(self):
        global sound_flg
        global vibr_flg
        global mes_flg
        global start
        global end
        global vibr_start
        global vibr_end

        ret, frame = self.cap.read()
        image, verd = cds.CreateLiveDetectorImage(frame, self.pose, self.mp_pose, self.mp_drawing, self.classificator)
        image = QImage(image, image.shape[1], image.shape[0],
                       image.strides[0], QImage.Format_RGB888)
        vibr_end = time.time()
        if verd == 0:
            end = time.time()
        else:
            start = time.time()

        if end - start > 5:
            start = time.time()
            if mes_flg:
                notification.notify(message='Осанка нарушена. Выпрямитесь!', app_name='Corrector')
            if sound_flg:
                chime.success()
            if vibr_flg:
                vibr_start = time.time()
                print('vkl')
                # s.write(str.encode('1'))
        if vibr_end - vibr_start < 0.8 and vibr_end - vibr_start > 0.6:
            print(vibr_end - vibr_start)
            # s.write(str.encode('0'))

        self.ui.image_label.setPixmap(QPixmap.fromImage(image))

    def stop_button(self):
        if self.timer.isActive():
            self.timer.stop()
        self.ui.text_label.setPixmap(QPixmap(u"UI/5.png"))
        self.ui.image_label_2.setText(QCoreApplication.translate("Detector",
                                                                 u"<html><head/><body><p align=\"center\">\u041f\u0430\u043c\u044f\u0442\u043a\u0430</p></body></html>",
                                                                 None))

    def finish_creating(self):
        if self.timer.isActive():
            self.timer.stop()
        global pointscoords
        feat = cds.CreateFeat(pointscoords)
        cds.WriteToExcel(pointscoords, feat)
        #uadd.UploadToDrive()
        self.ui.image_label_2.setText(QCoreApplication.translate("Detector",
                                                                 u"<html><head/><body><p align=\"center\">\u041f\u0430\u043c\u044f\u0442\u043a\u0430</p></body></html>",
                                                                 None))
        self.ui.text_label.setPixmap(QPixmap(u"UI/5.png"))

    def start_creating(self):
        self.ui.text_label.setText(QCoreApplication.translate("Detector",
                                                              u"<html><head/><body><p align=\"center\"></p></body></html>",
                                                              None))
        global start
        global end
        global pointscoords
        pointscoords = []
        start = time.time()
        end = time.time()
        self.timer = QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.display_creating_video_stream)
        self.cap = cv2.VideoCapture(0)
        self.timer.start()

    def display_creating_video_stream(self):
        global start
        global end
        global pointscoords
        ret, frame = self.cap.read()
        nowsec = end - start
        image, res = cds.CreateLiveCreatingImage(frame, self.pose, self.mp_pose, self.mp_drawing,
                                                 self.classificator, nowsec)
        pointscoords.append(res)
        image = QImage(image, image.shape[1], image.shape[0],
                       image.strides[0], QImage.Format_RGB888)
        end = time.time()
        self.ui.image_label_2.setPixmap(QPixmap.fromImage(image))

        if end - start > 80:
            self.finish_creating()
