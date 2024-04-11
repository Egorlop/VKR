import time
import mediapipe as mp
import cv2
import os
from pandas import DataFrame
import numpy as np
import pandas as pd
import datetime
import GoogleDrive.UploadAndDownloadDrive as uadd
from os import listdir
from os.path import isfile, join
import PostureDetecting.CreateDataSet as cds

def CreateDataSet(mp_drawing,mp_pose,mp_holistic,cap,type):
    pointscoords = []
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        now1 = datetime.datetime.now()
        while cap.isOpened():
            ret, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
                nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                            landmarks[mp_pose.PoseLandmark.NOSE.value].y,
                                 landmarks[mp_pose.PoseLandmark.NOSE.value].z]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
            except:
                pass
            now2 = datetime.datetime.now()
            nowsec = (now2 - now1)
            cv2.rectangle(image, (0, 0), (600, 73), (250, 250, 250), -1)
            cv2.putText(image, 'CreateDataSet' + ": "+ str(nowsec.seconds)+' sec', (45, 41),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )
            pointscoords.append(left_shoulder+right_shoulder+nose)
            cv2.imshow('Mediapipe Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    return pointscoords,mp_drawing,mp_pose,mp_holistic,cap

def WriteToExcel(pointscoords,feat):
    transposepoints = np.transpose(pointscoords)
    df = DataFrame({'X1': transposepoints[0], 'Y1': transposepoints[1], 'Z1': transposepoints[2],
                    'X2': transposepoints[3], 'Y2': transposepoints[4], 'Z2': transposepoints[5],
                    'X3': transposepoints[6], 'Y3': transposepoints[7], 'Z3': transposepoints[8],
                    'LABEL': feat})
    df.drop(df.index[(len(df)//2)-6:(len(df)//2)+6], inplace=True)
    df.drop(df.index[(len(df) // 4) - 3:(len(df) // 4) + 9], inplace=True)
    df.to_excel('D:\\pythonProject\\datasets\\CustomDataset.xlsx', sheet_name='sheet1', index=False)
    time.sleep(0.2)


def CreateFeat(pointscoords):
    feat = [1 for i in range(len(pointscoords) // 4)] + [0 for i in range(len(pointscoords) // 4)] +[1 for i in range(len(pointscoords) // 4)] + [0 for i in range(len(pointscoords) // 4)]
    for i in range(len(pointscoords) % 4):
        feat.append(0)
    print(len(pointscoords),len(feat))
    return feat


def ReadFromExcel(type):
    print(type)
    if type == 'All':
        uadd
        onlyfiles = [f for f in listdir('D:\\pythonProject\\datasets') if
                     isfile(join('D:\\pythonProject\\datasets', f))]
        for i in range(len(onlyfiles)):
            if i==0:
                data = pd.read_excel(f'D:\\pythonProject\\datasets\\{onlyfiles[i]}')
            else:
                data = pd.concat([data,pd.read_excel(f'D:\\pythonProject\\datasets\\{onlyfiles[i]}')])
        fromxl = []
        feat = data['LABEL'].values
        for i in range(1, 4):
            for sym in ['X', 'Y']:
                fromxl.append(data[sym + str(i)].values)
        fromxl = np.transpose(fromxl)
    else:
        data = pd.read_excel(f'D:\\pythonProject\\datasets\\{type}DataSet.xlsx')
        print(data)
        fromxl = []
        feat=data['LABEL'].values
        for i in range(1,4):
            for sym in ['X', 'Y']:
                fromxl.append(data[sym + str(i)].values)
        fromxl = np.transpose(fromxl)

    return fromxl,data,feat

def LiveTest(classificator):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(0)
    tests = []
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            shape = image.shape
            try:
                landmarks = results.pose_landmarks.landmark
                # Get coordinates
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
                                , landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
                                 #,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z
                                 ]
                nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x
                        ,landmarks[mp_pose.PoseLandmark.NOSE.value].y
                        #,landmarks[mp_pose.PoseLandmark.NOSE.value].z
                        ]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
                                  ,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
                                  #,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z
                                  ]
            except:
                pass

            test = [left_shoulder + right_shoulder + nose]
            tests.append(left_shoulder + right_shoulder + nose)

            verd = classificator.predict(test)
            if verd == 1:
                text = 'Correct position'
                color = (0, 230, 0)
            else:
                text = 'Wrong position. Straighten up!'
                color = (0, 0, 230)
            cv2.rectangle(image, (0, 0), (640, 73), color, -1)

            cv2.putText(image, text, (45, 41),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            cv2.imshow('Mediapipe Feed', image)
            time.sleep(0.1)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

def CreateLiveDetectorImage(frame,pose,mp_pose,mp_drawing,classificator):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image.flags.writeable = True
    try:
        landmarks = results.pose_landmarks.landmark
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
            , landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
                         ]
        nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x
            , landmarks[mp_pose.PoseLandmark.NOSE.value].y
                ]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
            , landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
                          ]
        test = [left_shoulder + right_shoulder + nose]
        verd = classificator.predict(test)
        if verd == 1:
            text = 'Correct position'
            color = (246,205,60)
            pos = (187, 39)
        else:
            text = 'Wrong position. Straighten up!'
            #color = (63,114,175)
            color = (202,187,233)
            pos = (90,39)
        cv2.rectangle(image, (0, 0), (640, 60), color, -1)
        cv2.putText(image, text, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        #                           mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
        #                           mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        #                           )
    except:
        text = 'Can`t see you'
        color = (200, 0, 0)
        cv2.rectangle(image, (0, 0), (640, 73), color, -1)
        cv2.putText(image, text, (45, 41),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return image,verd

def CreateLiveCreatingImage(frame,pose,mp_pose,mp_drawing,classificator,nowsec):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image.flags.writeable = True
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    landmarks = results.pose_landmarks.landmark
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
    nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                landmarks[mp_pose.PoseLandmark.NOSE.value].y,
                landmarks[mp_pose.PoseLandmark.NOSE.value].z]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
    if  round(nowsec,1)//10 == 0 or round(nowsec,1)//10 == 2:
        text = f'Stage {int((round(nowsec,1)//10))+1} - Take correct position'
        color = (82,195,82)
        pos = (187, 39)
    else:
        text = f'Stage {int((round(nowsec,1)//10))+1} - Take uncorrected position'
        color = (235, 82, 82)
        pos = (90, 39)
    cv2.rectangle(image, (0, 0), (640, 73), color, -1)
    cv2.putText(image, text , (20, 43),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, str(round(nowsec,1)) + ' sec', (520, 43),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    res = (left_shoulder + right_shoulder + nose)

    return image, res