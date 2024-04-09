import os
import GoogleDrive.UploadAndDownloadDrive as uadd
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import QMainWindow
from posture_detector import Ui_MainWindow
import PostureDetecting.Classificators as cl
import Metrics.KnnMetrics as km
import PostureDetecting.CreateDataSet as cds
from sklearn.model_selection import train_test_split
import mediapipe as mp
import cv2
def NewUserDataset(user='all'):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(0)
    datasetcoords,mp_drawing,mp_pose,mp_holistic,cap = cds.CreateDataSet(mp_drawing,mp_pose,mp_holistic,cap,user)
    feat = cds.CreateFeat(datasetcoords,user)
    cds.WriteToExcel(datasetcoords,feat,user)
    uadd.UploadToDrive()
    os.remove('D:\\pythonProject\\datasets\\CustomDataset.xlsx')
class PostureDetector(QMainWindow):
    def __init__(self):
        datasetxl, datasetpd, datasetfeat = cds.ReadFromExcel('All')
        testdata, testfeat, traindata, validatedata, trainfeat, validatefeat = SplitDataset(datasetxl, datasetfeat)
        self.classificator = PickClassificator(testdata, testfeat, traindata, trainfeat, validatedata, validatefeat)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        super(PostureDetector, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.timer = QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.display_video_stream)
        self.cap = cv2.VideoCapture(0)
        self.timer.start()

    def display_video_stream(self):
        ret, frame = self.cap.read()
        image = cds.CreateLiveImage(frame,self.pose,self.mp_pose,self.mp_drawing,self.classificator)
        image = QImage(image, image.shape[1], image.shape[0],
                       image.strides[0], QImage.Format_RGB888)
        self.ui.image_label.setPixmap(QPixmap.fromImage(image))