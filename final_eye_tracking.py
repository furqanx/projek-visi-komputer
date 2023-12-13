from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import pygame
import time
import dlib
import cv2

# Inisialisasi pygame
pygame.mixer.init()

# Memuat suara alarm
alarm_sound = pygame.mixer.Sound('alarm/mixkit-classic-alarm-995.wav')

# Fungsi untuk menghitung EAR (Eye Aspect Ratio)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Fungsi untuk mendeteksi warna mata (hitam dan putih)
def detect_eye_color(roi_color):
    # Tentukan threshold untuk warna hitam pada gambar
    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([30, 30, 30], dtype=np.uint8)
    # Tentukan threshold untuk warna putih pada gambar
    lower_white = np.array([200, 200, 200], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)

    # Ubah gambar ke ruang warna HSV
    hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)

    # Buat mask untuk warna hitam
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    # Buat mask untuk warna putih
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Hitung jumlah piksel hitam dan putih dalam mask
    black_pixel_count = cv2.countNonZero(mask_black)
    white_pixel_count = cv2.countNonZero(mask_white)

    # Jumlah total piksel yang dihitung
    total_pixel_count = black_pixel_count + white_pixel_count

    return total_pixel_count

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default="",
    help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0,
    help="index of webcam on system")
args = vars(ap.parse_args())

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48
# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
ALARM_ON = False

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

# loop over frames from the video stream
while True:
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            # if the eyes were closed for a sufficient number of
            # then sound the alarm
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                # if the alarm is not on, turn it on
                if not ALARM_ON:
                    ALARM_ON = True
                    # check to see if an alarm file was supplied,
                    # and if so, start a thread to have the alarm
                    # sound played in the background
                    if args["alarm"] != "":
                        t = Thread(target=sound_alarm,
                            args=(args["alarm"],))
                        t.deamon = True
                        t.start()
                # draw an alarm on the frame
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # otherwise, the eye aspect ratio is not below the blink
        # threshold, so reset the counter and alarm
        else:
            COUNTER = 0
            ALARM_ON = False

        # draw the computed eye aspect ratio on the frame to help
        # with debugging and setting the correct eye aspect ratio
        # thresholds and frame counters
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Dapatkan region of interest (ROI) untuk mata kiri dan kanan
        left_eye_roi = frame[shape[37][1]:shape[40][1], shape[36][0]:shape[39][0]]
        right_eye_roi = frame[shape[43][1]:shape[46][1], shape[42][0]:shape[45][0]]

        # Deteksi warna mata untuk mata kiri dan kanan
        left_eye_color = detect_eye_color(left_eye_roi)
        right_eye_color = detect_eye_color(right_eye_roi)

        # Menampilkan nilai left_eye_color dan right_eye_color di video
        cv2.putText(frame, f'Left Eye Color: {left_eye_color}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f'Right Eye Color: {right_eye_color}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Cek apakah EAR rata-rata kurang dari 0.2 dan mata terdeteksi (hitam atau putih)
        if ear < 0.2 and (left_eye_color > 0 or right_eye_color > 0):
            # Hidupkan alarm jika belum menyala dan telah melebihi 5 detik
            if not ALARM_ON and (time.time() - start_time) >= delay_time:
                ALARM_ON = True
                pygame.mixer.Channel(0).play(alarm_sound, loops=-1)  # -1 untuk memainkan suara secara terus menerus
                start_time = time.time()
        else:
            # Matikan alarm jika sedang menyala
            if ALARM_ON:
                ALARM_ON = False
                pygame.mixer.Channel(0).stop()
                start_time = time.time()  # Setel ulang start_time jika kondisi tidak terpenuhi

    # Menampilkan gambar
    cv2.imshow("Output", frame)

    # Menunggu tombol ESC untuk keluar
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
