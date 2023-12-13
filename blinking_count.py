# Necessary imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
import dlib

# Initializing the face detector and facial landmark predictor 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Creating a VidoCapture and VideoWriter object
cap=cv2.VideoCapture("0")
fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
# out = cv2.VideoWriter('output.mp4', fourcc, 29, (1080,1920))
font=cv2.FONT_HERSHEY_SIMPLEX


# Defining the mid-point
def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)
# Defining the Euclidean distance
def euclidean_distance(leftx,lefty, rightx, righty):
    return np.sqrt((leftx-rightx)**2 +(lefty-righty)**2)


# Defining the eye aspect ratio
def get_EAR(eye_points, facial_landmarks):
# Defining the left point of the eye   
    left_point = [facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y]
# Defining the right point of the eye   
    right_point = [facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y]
# Defining the top mid-point of the eye    
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
# Defining the bottom mid-point of the eye   
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
# Drawing horizontal and vertical line       
    hor_line = cv2.line(frame, (left_point[0], left_point[1]), (right_point[0], right_point[1]), (255, 0, 0), 3)
    ver_line = cv2.line(frame, (center_top[0], center_top[1]),(center_bottom[0], center_bottom[1]), (255, 0, 0), 3)
# Calculating length of the horizontal and vertical line    
    hor_line_lenght = euclidean_distance(left_point[0], left_point[1], right_point[0], right_point[1])
    ver_line_lenght = euclidean_distance(center_top[0], center_top[1], center_bottom[0], center_bottom[1])
# Calculating eye aspect ratio     
    EAR = ver_line_lenght / hor_line_lenght
    return EAR

# Creating a list eye_blink_signal
eye_blink_signal=[]
# Creating an object blink_ counter
blink_counter = 0
previous_ratio = 100
# Creating a while loop
while True:
  ret, frame = cap.read() 
  if ret == False:
    break
# Converting a color frame into a grayscale frame  
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# Creating an object in which we will sore detected faces  
  faces = detector(gray)
  for face in faces:
    x, y = face.left(), face.top()
    x1, y1 = face.right(), face.bottom()
# Creating an object in which we will sore detected facial landmarks
    landmarks = predictor(gray, face)
# Calculating left eye aspect ratio    
    left_eye_ratio = get_EAR([36, 37, 38, 39, 40, 41], landmarks)
# Calculating right eye aspect ratio  
    right_eye_ratio = get_EAR([42, 43, 44, 45, 46, 47], landmarks)
# Calculating aspect ratio for both eyes  
    blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
# Rounding blinking_ratio on two decimal places   
    blinking_ratio_1 = blinking_ratio * 100
    blinking_ratio_2 = np.round(blinking_ratio_1)
    blinking_ratio_rounded = blinking_ratio_2 / 100
# Appending blinking ratio to a list eye_blink_signal
    eye_blink_signal.append(blinking_ratio)
    if blinking_ratio < 0.20:
      if previous_ratio > 0.20:
        blink_counter = blink_counter + 1
 # Displaying blink counter and blinking ratio in our output video      
    
    previous_ratio = blinking_ratio
  
  cv2.putText(frame, str(blink_counter), (30, 50), font, 2, (0, 0, 255),5)
  cv2.putText(frame, str(blinking_ratio_rounded), (900, 50), font, 2, (0, 0, 255),5)
  


