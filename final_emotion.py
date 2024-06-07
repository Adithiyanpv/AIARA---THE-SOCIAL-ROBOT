import logging

# Set the logging level to suppress the detailed output
logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.WARNING)
import warnings
# Suppress the warnings
warnings.filterwarnings("ignore")
from os import environ
from logging import getLogger,ERROR

# Silence the TensorFlow deprecation warning
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
getLogger('tensorflow').setLevel(ERROR)


from sys import argv,exit
import cv2
from deepface import DeepFace
import numpy as np
from ultralytics import YOLO
import google.generativeai as genai
import os
GOOGLE_API_KEY="AIzaSyA6xxFQMg3pvz8cWGQcLOTg5jCrz23Ao7w"


# def gemini(s,GOOGLE_API_KEY):
#     name, emotion, prompt = s.split('|')
#     print(s)
#     os.getenv('GOOGLE_API_KEY')
#     genai.configure(api_key=GOOGLE_API_KEY)
#     model = genai.GenerativeModel('gemini-pro')
#     if emotion == "NONE" and name == "NONE":
#         promptmod="Your role: a social robot(AIARA), software who responds to my query, query:" +prompt
#     elif name == "NONE":
#         promptmod="Your role: a social robot(AIARA), software who responds to my query with context of my emotion" +"\n my emotion:" + emotion + "query:"+ prompt
#     elif emotion == "NONE":
#         promptmod="Your role: a social robot(AIARA), software who responds to my query with context of my \n my name:"+ name  + "query:"+ prompt
#     else:
#         promptmod="Your role: a social robot(AIARA), software who responds to my query with context of my name and emotion\n my name:"+ name + ",my emotion:" + emotion + "query:"+ prompt
#     print(promptmod)
#     response=model.generate_content(promptmod)
#     return response.text

# def gemini(s,GOOGLE_API_KEY):
#     name, emotion, prompt = s.split('|')
#     print(s)
#     os.getenv('GOOGLE_API_KEY')
#     genai.configure(api_key=GOOGLE_API_KEY)
#     model = genai.GenerativeModel('gemini-pro')
#     if emotion == "neutral" or "NONE":
#         promptmod="Your role:A friendly talking social robot; greet me before answering; My name is "+name+"; Answer in short unless explicitly told. Output the emotion of your response in the first line in one word('happy','neutral','sad','laughing') then a single vertical bar '|' then your answer, no spaces. Query:"+prompt
#     else:
#         promptmod="Your role:A friendly talking social robot; greet me before answering; My name is"+name+" ;I am sad; Answer in short unless explicitly told. Output the emotion of your response in the first line in one word('sad') then a '|' actual response from next line; Query:"+prompt
#     # print(promptmod)
#     response=model.generate_content(promptmod)
#     # print("---------------------------"+response.text+"---------------------------")
#     return response.text

def gemini(s,GOOGLE_API_KEY):
    name, emotion, prompt = s.split('|')
    # print(s)
    os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    if emotion == "neutral" or "NONE":
        promptmod='''Your role: A friendly social chatbot. Your response format: Emotion of your answer-either happy/sad/neutral/laughing(laughing only when telling jokes) 
+ greeting/congratulating/comforting/consoling + your generated response for query. 
First word of your reply should just contain the hypothetical emotional state of your response. The response should be short and sweet, unless explicitly mentioned in the query.
My name:'''+name+""". my query:"""+prompt
    else:
        promptmod='''Your role: A friendly social chatbot. 
Your response format: emotion of your response 
+ greeting/congratulating/comforting/consoling + your generated response for query. 
First word of your reply should just contain the hypothetical emotional state of your response. The response should be short and sweet, unless explicitly mentioned in the query.
My name:'''+name+""". my emotion:sad. my query:"""+prompt    
    response=model.generate_content(promptmod)
    s=response.text
    index = s.find("-")

    # Replace "-" with "|"
    s = s[:index] + "|" + s[index+1:]
    return s

def voiceProcess(video_path):
    import whisper
    model = whisper.load_model("base")
    result = model.transcribe(video_path)
    return result['text']

def process_video(video_path):
    from os import listdir
    names = listdir("./final/data/test")
    # print(names)
    class_counts = {}
    model = YOLO("./final/runs/classify/train6/weights/best.pt")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(video_path)
    minutes = 0
    seconds = 0
    fps = 0.5
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break  # Break the loop when the video ends
        t_msec = 1000*fps*(minutes*60 + seconds)
        cap.set(cv2.CAP_PROP_POS_MSEC, t_msec)
        # Convert frame to grayscale
        seconds += 1
        if seconds > 60:
            seconds = 0
            minutes += 1


        predictions = model(frame,verbose=False)
        predicted_class = names[predictions[0].probs.top1]  # Assuming predictions contain class probabilities
        class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Initialize a list to store the detected emotions
        emotions = []

        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract the face ROI (Region of Interest)
            face_roi = frame[y:y + h, x:x + w]

            # Perform emotion analysis on the face ROI
            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                emotions.append(emotion)
            except:
                # Ignore any errors that occur during emotion analysis
                pass
    #
    # Print the dominant emotion and the list of detected emotions
    max_count_class = "NONE"
    dominant_emotion = "NONE"
    try:
        max_count_class = max(class_counts, key=class_counts.get)
        dominant_emotion = max(set(emotions), key=emotions.count)
        # print(f"Detected Emotion: {dominant_emotion} {emotions}")
    except:
        pass
    # Release the capture
    cap.release()    
    return max_count_class,dominant_emotion

if __name__ == "__main__":
    if len(argv) < 2:
        print("Usage: python processing.py <video_path>")
        exit(1)

    # video_path = "./final/tests/testing4.webm"
    video_path = argv[1]
    processed_output = process_video(video_path)
    voice_process  = voiceProcess(video_path)
    # print(voice_process)
    kk =f"{processed_output[0]}|{processed_output[1]}|{voice_process}"
    final_ = gemini(kk,GOOGLE_API_KEY)
    kk = f"{voice_process}|{final_}"
    print(kk)