import time
import cv2
import PostureDetecting.CreateDataSet as cds
import PostureDetecting.Classificators as cl
import Metrics.WriteScatter as ws
import Metrics.KnnMetrics as km
import mediapipe as mp
from sklearn.model_selection import train_test_split


def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(0)

    user = 'All'
    #datasetcoords,mp_drawing,mp_pose,mp_holistic,cap = cds.CreateDataSet(mp_drawing,mp_pose,mp_holistic,cap,user)
    #feat = cds.CreateFeat(datasetcoords,user)
    #cds.WriteToExcel(datasetcoords,feat,user)
    datasetxl,datasetpd,datasetfeat = cds.ReadFromExcel(user)


    x, testdata, y, testfeat = train_test_split(datasetxl, datasetfeat, test_size=0.2, train_size=0.8)
    traindata, validatedata, trainfeat, validatefeat = train_test_split(x, y, test_size=0.25, train_size=0.75)
    #print(len(testdata), len(traindata), len(validatedata))
    #print(len(testfeat), len(trainfeat), len(validatefeat))
    bestRNumber = km.GetBetterRNumber(testdata,testfeat,traindata,trainfeat,validatedata,validatefeat)
    #bestKNumber = km.GetBetterKNumber(testdata,testfeat,traindata,trainfeat,validatedata,validatefeat)
    #neiClassificator = cl.KNeighbors(traindata, trainfeat, bestKNumber)
    randomForest = cl.RForest(traindata, trainfeat, bestRNumber)
    cds.LiveTest(randomForest,mp_drawing,mp_pose,mp_holistic,cap)

    #ws.WriteScatter()
    #km.KnnMeshgrid()



main()