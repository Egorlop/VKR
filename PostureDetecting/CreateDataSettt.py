import cv2
import time



def CreateTestDataSet(mp_drawing,mp_pose,mp_holistic,cap):
    pointscoords = []
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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

            cv2.rectangle(image, (0, 0), (300, 73), (250, 250, 250), -1)
            cv2.putText(image, 'Obychaem', (45, 41),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            pointscoords.append(left_shoulder+right_shoulder+nose)
            time.sleep(0.2)
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    return pointscoords,mp_drawing,mp_pose,mp_holistic,cap