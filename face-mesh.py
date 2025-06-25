import time
import cv2 as cv
import mediapipe as mp

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()

mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(thickness = 1, circle_radius = 1)

cap = cv.VideoCapture(0)

cTime = 0
pTime = 0

while True:
    succes, frame = cap.read()

    if not succes:
        break

    else:
        frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        results = faceMesh.process(frameRGB)

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime 
        
        cv.putText(frame, "FPS: " + str(int(fps)), (100, 100), cv.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)

        cv.imshow("face-mesh", frame)

    quit = cv.waitKey(1)
    if quit != -1:
        break