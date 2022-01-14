import encodings
from unittest import suite
import face_recognition
import os
import cv2
import shutil
from pathlib import Path

IMAGE_DIR = './images'
RESULT_DIR = './results'
TOLERANE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'cnn'

print("processing images")
faces = []
face_names = []
fact_count = 0

for filename in os.listdir(IMAGE_DIR):
    src_path=f"{IMAGE_DIR}/{filename}"
    image = face_recognition.load_image_file(src_path)
    face_locations =  face_recognition.face_locations(image, model=MODEL)
    face_encodings = face_recognition.face_encodings(image, face_locations  )
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for encoding, location in zip(face_encodings, face_locations):
        results = face_recognition.compare_faces(faces, encoding, TOLERANE)
        match = None
        for index, result in enumerate(results):
            if result:
                match = face_names[index]
                dst_path = f"{RESULT_DIR}/{match}"
                # Check if director presented for current face
                Path(dst_path).mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
                print(f"Match found for {filename}: {match}")
                top_left = (location[3], location[0])
                bottom_right = (location[1], location[2])
                color = [0,255,0]
                cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

                top_left = (location[3], location[2])
                bottom_right = (location[1], location[2]+22)
                cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
                cv2.putText(image, match, (location[3]+10, location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), FONT_THICKNESS)


            else:
                faces.append(encoding)
                face_names.append(f"face_{fact_count:03d}")
                fact_count+=1
    
    cv2.imshow(filename, image)
    cv2.waitKey(10000)
    # cv2.destroyWindow(filename)
