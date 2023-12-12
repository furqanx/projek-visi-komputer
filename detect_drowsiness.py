from imutils import face_utils
import dlib
import cv2
import pygame

# Inisialisasi pygame
pygame.mixer.init()

# Memuat suara alarm
alarm_sound = pygame.mixer.Sound('alarm/mixkit-classic-alarm-995.wav')

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)

# Inisialisasi status mata terakhir
last_eye_status = "open"

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
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        # Menghitung EAR untuk mata kiri dan kanan
        left_ear = eye_aspect_ratio(shape[left_eye_indices])
        right_ear = eye_aspect_ratio(shape[right_eye_indices])

        # Menghitung EAR rata-rata untuk kedua mata
        ear_avg = (left_ear + right_ear) / 2.0

        # Menentukan status mata berdasarkan threshold
        eye_status = "open" if ear_avg > 0.25 else "closed"

        # Memainkan alarm jika mata terbuka berubah menjadi tertutup
        if last_eye_status == "open" and eye_status == "closed":
            pygame.mixer.Sound.play(alarm_sound)

        # Memperbarui status mata terakhir
        last_eye_status = eye_status

    # Menampilkan gambar
    cv2.imshow("Output", image)

    # Menunggu tombol ESC untuk keluar
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
