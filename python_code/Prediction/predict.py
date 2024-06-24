import os.path

import cv2
import numpy as np
import tkinter as tk

from keras.models import load_model
from keras.utils import img_to_array
from tkinter.filedialog import askopenfilename, asksaveasfilename

root = tk.Tk()
root.withdraw()

def detector(frame, model, faces):
    for (x, y, w, h) in faces:
        face_image = frame[y:y + h, x:x + w]

        pred_image = cv2.resize(face_image, (35, 35))
        pred_image = cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB)
        pred_image = img_to_array(pred_image) / 255.0
        pred_image = np.expand_dims(pred_image, axis=0)

        prediction = model.predict(pred_image).argmax(axis=1)
        if np.round(prediction[0]) == 0:
            prediction_label = 'Using Mask'
            color = (0, 255, 0)
        else:
            prediction_label = 'Not Using Mask'
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), color, 2)
        cv2.putText(frame, f'{prediction_label}', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return frame

model_file = askopenfilename(title="Select model")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model(model_file)

while True:
    if not os.path.exists('results'):
        os.makedirs('results')

    predict_option = input("Do you want to predict image or video ? ")
    if (predict_option == "image") or (predict_option == "Image") or (predict_option == "IMAGE"):
        no_mask_path = askopenfilename(title="Select not using mask image")
        use_mask_path = askopenfilename(title="Select using mask image")

        no_mask_image = cv2.imread(no_mask_path)
        use_mask_image = cv2.imread(use_mask_path)
        stacked_image = cv2.hconcat([no_mask_image, use_mask_image])

        faces = face_cascade.detectMultiScale(stacked_image, 1.1, 10)
        frame = detector(stacked_image, model, faces)

        cv2.imwrite('results/img_result.jpg', frame)

        cv2.imshow('Face Mask Detection', frame)
        cv2.waitKey(0)

        break
    if (predict_option == "video") or (predict_option == "Video") or (predict_option == "VIDEO"):
        input_vid = askopenfilename(title="Select input video")

        cap = cv2.VideoCapture(input_vid)
        output_vid = asksaveasfilename(title="Save as", filetypes=[("Video files", "*.mp4")])

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writter = cv2.VideoWriter(output_vid, fourcc, fps, frame_size)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = face_cascade.detectMultiScale(frame, 1.9, 5)
            frame = detector(frame, model, faces)
            video_writter.write(frame)

            cv2.imshow('Face Mask Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        video_writter.release()
        cv2.destroyAllWindows()

        break
    else:
        print('The answer must be yes/no')