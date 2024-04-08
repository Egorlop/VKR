import Metrics.WriteScatter as ws
import cv2
import PostureDetecting.CreateDataSet as cds
import PostureDetecting.Classificators as cl
import Metrics.KnnMetrics as km
import GoogleDrive.UploadAndDownloadDrive as uadd
import mediapipe as mp
import os
from sklearn.model_selection import train_test_split
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
def SplitDataset(datasetxl, datasetfeat):
    x, testdata, y, testfeat = train_test_split(datasetxl, datasetfeat, test_size=0.2, train_size=0.8)
    traindata, validatedata, trainfeat, validatefeat = train_test_split(x, y, test_size=0.25, train_size=0.75)
    return testdata,testfeat, traindata, validatedata, trainfeat, validatefeat
def PickClassificator(testdata,testfeat,traindata,trainfeat,validatedata,validatefeat):
    bestRNumber = km.GetBetterRNumber(testdata,testfeat,traindata,trainfeat,validatedata,validatefeat)
    bestKNumber = km.GetBetterKNumber(testdata,testfeat,traindata,trainfeat,validatedata,validatefeat)
    # if bestRNumber > bestKNumber:
    #     classificator = cl.RForest(traindata, trainfeat, bestKNumber)
    # else:
    #     classificator = cl.KNeighbors(traindata, trainfeat, bestKNumber)
    return cl.KNeighbors(traindata, trainfeat, bestKNumber)
def main():
    #NewUserDataset()
    #uadd.DownloadFromDrive()
    datasetxl, datasetpd, datasetfeat = cds.ReadFromExcel('All')
    testdata,testfeat, traindata, validatedata, trainfeat, validatefeat = SplitDataset(datasetxl, datasetfeat)
    classificator=PickClassificator(testdata,testfeat,traindata,trainfeat,validatedata,validatefeat)
    cds.LiveTest(classificator)
    #ws.WriteScatter(datasetpd)
    #km.KnnMeshgrid()

main()