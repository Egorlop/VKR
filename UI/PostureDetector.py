from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import QMainWindow
#from UI.posture_detector import Ui_Detector
from UI.ui_main import Ui_Detector
import PostureDetecting.Classificators as cl
import Metrics.KnnMetrics as km
import PostureDetecting.CreateDataSet as cds
from sklearn.model_selection import train_test_split
import mediapipe as mp
import cv2
import time
from plyer import notification
import chime

start = time.time()
end = time.time()
sound_flg = 0
vibr_flg = 0
mes_flg = 0
def SplitDataset(datasetxl, datasetfeat):
    x, testdata, y, testfeat = train_test_split(datasetxl, datasetfeat, test_size=0.2, train_size=0.8)
    traindata, validatedata, trainfeat, validatefeat = train_test_split(x, y, test_size=0.25, train_size=0.75)
    return testdata,testfeat, traindata, validatedata, trainfeat, validatefeat

def PickClassificator(testdata,testfeat,traindata,trainfeat,validatedata,validatefeat):
    bestRNumber = km.GetBetterRNumber(testdata,testfeat,traindata,trainfeat,validatedata,validatefeat)
    bestKNumber = km.GetBetterKNumber(testdata,testfeat,traindata,trainfeat,validatedata,validatefeat)
    if bestRNumber > bestKNumber:
         classificator = cl.RForest(traindata, trainfeat, bestKNumber)
    else:
         classificator = cl.KNeighbors(traindata, trainfeat, bestKNumber)
    return classificator
class PostureDetector(QMainWindow):
    def __init__(self):
        datasetxl, datasetpd, datasetfeat = cds.ReadFromExcel('All')
        testdata, testfeat, traindata, validatedata, trainfeat, validatefeat = SplitDataset(datasetxl, datasetfeat)
        self.classificator = PickClassificator(testdata, testfeat, traindata, trainfeat, validatedata, validatefeat)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        QMainWindow.__init__(self)
        self.ui = Ui_Detector()
        self.ui.setupUi(self)

        self.ui.btn_page_3.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page_3))
        self.ui.btn_page_3.clicked.connect(self.finish)

        self.ui.btn_page_1.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page_1))
        self.ui.btn_page_1.clicked.connect(self.finish)

        self.ui.btn_page_2.clicked.connect(self.start)
        self.ui.btn_page_2.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page_2))

        state = self.ui.checkBoxSound.stateChanged.connect(self.sound_checkbox_changed)
        self.ui.checkBoxVibr.stateChanged.connect(self.vibr_checkbox_changed)
        self.ui.checkBoxMes.stateChanged.connect(self.mes_checkbox_changed)




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
    def finish(self):
        if self.timer.isActive():
            self.timer.stop()
    def start(self):
        global start
        global end
        self.timer = QTimer()
        self.timer.setInterval(100)
        start = time.time()
        end = time.time()
        self.timer.timeout.connect(self.display_video_stream)
        self.cap = cv2.VideoCapture(0)
        self.timer.start()
    def display_video_stream(self):
        global sound_flg
        global vibr_flg
        global mes_flg
        global start
        global end
        ret, frame = self.cap.read()
        image,verd = cds.CreateLiveImage(frame,self.pose,self.mp_pose,self.mp_drawing,self.classificator)
        image = QImage(image, image.shape[1], image.shape[0],
                       image.strides[0], QImage.Format_RGB888)
        if verd == 0:
            end = time.time()
            res = start - end
        else:
            start = time.time()
        if end - start > 5:
            start = time.time()
            if mes_flg:
                notification.notify(message='Осанка нарушена. Выпрямитесь!', app_name='Corrector')
            if sound_flg:
                chime.success()
        self.ui.image_label.setPixmap(QPixmap.fromImage(image))