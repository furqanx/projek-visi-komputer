from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
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

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)



# Inisialisasi status alarm dan timestamp
alarm_on = False
start_time = 0
delay_time = 5  # Jeda 5 detik



while True:
    # Membaca frame dari webcam
    _, image = cap.read()

    # Konversi gambar ke skala abu-abu
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Mendapatkan wajah dalam gambar webcam
    rects = detector(gray, 0)

    # Untuk setiap wajah yang terdeteksi, temukan landmark.
    for (i, rect) in enumerate(rects):
        # Membuat prediksi dan mengubahnya menjadi array numpy
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Membuat list indeks landmark mata (mulai dari 36 hingga 47)
        left_eye_indices = list(range(36, 42))
        right_eye_indices = list(range(42, 48))

        # Menggambar landmark mata pada gambar
        for idx in left_eye_indices + right_eye_indices:
            x, y = shape[idx]

            left_points = shape[left_eye_indices]
            right_points = shape[right_eye_indices]
            cv2.polylines(image, [left_points], isClosed=True, color=(0, 255, 0), thickness=1)
            cv2.polylines(image, [right_points], isClosed=True, color=(0, 255, 0), thickness=1)

        # Menghitung EAR untuk mata kiri dan kanan
        left_ear = eye_aspect_ratio(shape[left_eye_indices])
        right_ear = eye_aspect_ratio(shape[right_eye_indices])

        # Menghitung EAR rata-rata untuk kedua mata
        ear_avg = (left_ear + right_ear) / 2.0

        # Menampilkan nilai EAR di sudut kanan atas
        ear_text = "EAR: {:.2f}".format(ear_avg)
        cv2.putText(image, ear_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



        # # Cek apakah EAR rata-rata kurang dari 0.2
        # if ear_avg < 0.2:
        #     # Hidupkan alarm jika belum menyala
        #     if not alarm_on:
        #         alarm_on = True
        #         pygame.mixer.Channel(0).play(alarm_sound, loops=-1)  # -1 untuk memainkan suara secara terus menerus
        # else:
        #     # Matikan alarm jika sedang menyala
        #     if alarm_on:
        #         alarm_on = False
        #         pygame.mixer.Channel(0).stop()



        # Cek apakah EAR rata-rata kurang dari 0.2
        # if ear_avg < 0.2:
        #     # Cek jeda sejak alarm dimatikan terakhir
        #     current_time = time.time()
        #     elapsed_time = current_time - last_alarm_time

        #     # Hidupkan alarm jika belum menyala dan jeda waktu telah terlampaui
        #     if not alarm_on and elapsed_time >= 5:
        #         alarm_on = True
        #         last_alarm_time = current_time
        #         pygame.mixer.Channel(0).play(alarm_sound, loops=-1)  # -1 untuk memainkan suara secara terus menerus
        # else:
        #     # Matikan alarm jika sedang menyala
        #     if alarm_on:
        #         alarm_on = False
        #         pygame.mixer.Channel(0).stop()



        # # Cek apakah EAR rata-rata kurang dari 0.2
        # if ear_avg < 0.2:
        #     # Hidupkan alarm jika belum menyala atau telah melebihi 5 detik
        #     # if not alarm_on or (time.time() - start_time) >= 5:
        #     if (time.time() - start_time) >= 5:
        #         alarm_on = True
        #         pygame.mixer.Channel(0).play(alarm_sound, loops=-1)  # -1 untuk memainkan suara secara terus menerus
        #         start_time = time.time()
        # else:
        #     # Matikan alarm jika sedang menyala
        #     if alarm_on:
        #         alarm_on = False
        #         pygame.mixer.Channel(0).stop()



        # Cek apakah EAR rata-rata kurang dari 0.2
        if ear_avg < 0.2:
            # Hidupkan alarm jika belum menyala dan telah melebihi 5 detik
            if not alarm_on and (time.time() - start_time) >= delay_time:
                alarm_on = True
                pygame.mixer.Channel(0).play(alarm_sound, loops=-1)  # -1 untuk memainkan suara secara terus menerus
                start_time = time.time()
        else:
            # Matikan alarm jika sedang menyala
            if alarm_on:
                alarm_on = False
                pygame.mixer.Channel(0).stop()
                start_time = time.time()  # Setel ulang start_time jika kondisi tidak terpenuhi






    # Menampilkan gambar
    cv2.imshow("Output", image)

    # Menunggu tombol ESC untuk keluar
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
