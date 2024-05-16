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
from PIL import ImageDraw, ImageFont, Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

fontpath = "seguisb.ttf"
#fontpath = "C:\\Windows\\Fonts\\Arial.ttf"
font = ImageFont.truetype(fontpath, 24)

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
    timer=str(datetime.datetime.now()).replace(' ', '_').replace(':', '-')[:-7]
    df.to_excel(f'D:\\pythonProject\\datasets\\{timer}_DataSet.xlsx', sheet_name='sheet1', index=False)
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
        validationfiles = [f for f in listdir('D:\\pythonProject\\datasets\\Validation') if
                     isfile(join('D:\\pythonProject\\datasets\\Validation', f))]
        for i in range(len(validationfiles)):
            if i==0:
                data = pd.read_excel(f'D:\\pythonProject\\datasets\\Validation\\{validationfiles[i]}')
            else:
                data = pd.concat([data,pd.read_excel(f'D:\\pythonProject\\datasets\\Validation\\{validationfiles[i]}')])
        validationfromxl = []
        validationfeat = data['LABEL'].values
        for i in range(1, 4):
            for sym in ['X', 'Y','Z']:
                validationfromxl.append(data[sym + str(i)].values)
        validationfromxl = np.transpose(validationfromxl)
        ################
        trainfiles = [f for f in listdir('D:\\pythonProject\\datasets\\Train') if
                     isfile(join('D:\\pythonProject\\datasets\\Train', f))]
        for i in range(len(trainfiles)):
            if i==0:
                data = pd.read_excel(f'D:\\pythonProject\\datasets\\Train\\{trainfiles[i]}')
            else:
                data = pd.concat([data,pd.read_excel(f'D:\\pythonProject\\datasets\\Train\\{trainfiles[i]}')])
        trainfromxl = []
        trainfeat = data['LABEL'].values
        for i in range(1, 4):
            for sym in ['X', 'Y','Z']:
                trainfromxl.append(data[sym + str(i)].values)
        trainfromxl = np.transpose(trainfromxl)
        ###########
        testfiles = [f for f in listdir('D:\\pythonProject\\datasets\\Test') if
                     isfile(join('D:\\pythonProject\\datasets\\Test', f))]
        for i in range(len(testfiles)):
            if i==0:
                data = pd.read_excel(f'D:\\pythonProject\\datasets\\Test\\{testfiles[i]}')
            else:
                data = pd.concat([data,pd.read_excel(f'D:\\pythonProject\\datasets\\Test\\{testfiles[i]}')])
        testfromxl = []
        testfeat = data['LABEL'].values
        for i in range(1, 4):
            for sym in ['X', 'Y','Z']:
                testfromxl.append(data[sym + str(i)].values)
        testfromxl = np.transpose(testfromxl)
    else:
        data = pd.read_excel(f'D:\\pythonProject\\datasets\\{type}DataSet.xlsx')
        print(data)
        fromxl = []
        feat=data['LABEL'].values
        for i in range(1,4):
            for sym in ['X', 'Y']:
                fromxl.append(data[sym + str(i)].values)
        fromxl = np.transpose(fromxl)

    return validationfromxl,validationfeat,trainfromxl,trainfeat,testfromxl,testfeat

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
                         ,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z
                         ]
        nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x
            , landmarks[mp_pose.PoseLandmark.NOSE.value].y
                ,landmarks[mp_pose.PoseLandmark.NOSE.value].z
                ]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
            , landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
                          ,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z
                          ]
        test = [left_shoulder + right_shoulder + nose]
        verd = classificator.predict(test)
        if verd == 1:
            text = 'Положение соответствует норме'
            color = (82,195,82)
            pos = (153, 13)
        else:
            text = 'Осанка нарушена. Выпрямитесь!'
            color = (235, 82, 82)
            pos = (153,13)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )
        cv2.rectangle(image, (0, 0), (640, 60), color, -1)
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        draw.text(pos, text, font=font)
        image = np.array(img_pil)
    except  Exception as e:
        print(e)
        text = 'Вас не видно.'
        verd=1
        color = (200, 0, 0)
        pos = (250, 13)
        cv2.rectangle(image, (0, 0), (640, 60), color, -1)
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        draw.text(pos, text, font=font)
        image = np.array(img_pil)
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
    if  round(nowsec,1)//20 == 0 or round(nowsec,1)//20 == 2:
        text = f'Стадия {int((round(nowsec, 1) // 10)) + 1} \nЗаймите правильное положение'
        color = (82,195,82)
        pos = (187, 39)
    else:
        text = f'Стадия {int((round(nowsec,1)//10))+1}\nЗаймите неправильное положение'
        color = (235, 82, 82)
        pos = (90, 39)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )
    cv2.rectangle(image, (0, 0), (640, 73), color, -1)
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    draw.text((30, 6), text, font=font)
    draw.text((545, 20), str(round(nowsec,1)) + ' sec', font=font)
    image = np.array(img_pil)

    res = (left_shoulder + right_shoulder + nose)

    return image, res

def CreateStatsImage(period):
    plt.rcParams['figure.figsize'] = [7, 5]
    plt.xlabel('День')
    plt.ylabel('Кол-во срабатываний/час')
    df = pd.read_csv('UI/Stats.csv', encoding='utf-8', delimiter=';')
    y = list(df['Кол-во срабатываний/час'])[0:90]
    print(y)
    y = [elem.replace(',', '.') for elem in y]
    y = [float(elem) for elem in y]
    if period == 'Месяц':
        plt.title(f'Динамика улучшения осанки\n за последний месяц')
        x = range(1,31)
        y = y[0:30]
        plt.plot(x, y, marker='*', color='y')
        plt.savefig('Month' + ".png", format="png")
    if period == 'Неделя':
        plt.title(f'Динамика улучшения осанки\n за последнюю неделю')
        x = range(1,8)
        y=y[23:30]
        plt.plot(x, y, marker='*', color='y')
        plt.savefig('Week' + ".png", format="png")
    if period == 'Квартал':
        plt.title(f'Динамика улучшения осанки\n за последний квартал')
        x = range(1,91)
        y = y[0:90]
        plt.plot(x, y, marker='*', color='y')
        plt.savefig('Quat' + ".png", format="png")
    plt.close()

