import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime


video_capture = cv2.VideoCapture('http://192.168.1.37:4747/video')
mehul_image = face_recognition.load_image_file("faces/mehul.jpg")
mehul_encoding = face_recognition.face_encodings(mehul_image)[0]  # 0 return the value(encoding) of first face from the faces

luffy_image = face_recognition.load_image_file("faces/luffy.png")
luffy_encoding = face_recognition.face_encodings(luffy_image)[0]

known_face_encodings = [mehul_encoding, luffy_encoding]
known_face_names = ["mehul", "luffy"]

# list of expected students
students = known_face_names.copy()

face_locations = []
face_encodings = []

# get the current date and time
now = datetime.now()
current_date = now.strftime("%d-%m-%Y")

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encodings in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encodings)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encodings)
        best_match_index = np.argmin(face_distance)

        if (matches[best_match_index]):
            name = known_face_names[best_match_index]
        '''else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            BottomLeftCornerOfText = (10, 100)
            fontScale = 1.2
            fontColor = (255, 255, 255)
            thickness = 3
            lineType = 2
            cv2.putText(rgb_small_frame, "not in data", BottomLeftCornerOfText,
                        font, fontScale, fontColor, thickness, lineType)'''

        # Add the text if a person is present
        if name in known_face_names:
            font = cv2.FONT_HERSHEY_SIMPLEX
            BottomLeftCornerOfText = (10, 100)
            fontScale = 1.2
            fontColor = (255,255,255)
            thickness = 3
            lineType = 2
            cv2.putText(rgb_small_frame, name + " Present", BottomLeftCornerOfText, font, fontScale,
                            fontColor, thickness, lineType)

        if name in students:
            students.remove(name)
            current_time = now.strftime("%H-%M-%S")
            lnwriter.writerow([name, current_time])

    cv2.imshow('frame', rgb_small_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()