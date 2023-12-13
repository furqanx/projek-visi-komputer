from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
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



        # Dapatkan region of interest (ROI) untuk mata kiri dan kanan
        left_eye_roi = image[shape[37][1]:shape[40][1], shape[36][0]:shape[39][0]]
        right_eye_roi = image[shape[43][1]:shape[46][1], shape[42][0]:shape[45][0]]

        # Deteksi warna mata untuk mata kiri dan kanan
        left_eye_color = detect_eye_color(left_eye_roi)
        right_eye_color = detect_eye_color(right_eye_roi)

        # Menampilkan nilai left_eye_color dan right_eye_color di video
        cv2.putText(image, f'Left Eye Color: {left_eye_color}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, f'Right Eye Color: {right_eye_color}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)




        # Cek apakah EAR rata-rata kurang dari 0.2 dan mata terdeteksi (hitam atau putih)
        if ear_avg < 0.2 and (left_eye_color > 0 or right_eye_color > 0):
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
