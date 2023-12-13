import cv2

# Fungsi untuk mendeteksi ROI pada pupil mata
def detect_pupil_roi(eye_roi):
    # Ubah gambar ke skala abu-abu
    gray_eye = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)

    # Lakukan thresholding pada gambar mata
    _, threshold_eye = cv2.threshold(gray_eye, 50, 255, cv2.THRESH_BINARY_INV)

    # Temukan kontur pada gambar threshold
    contours, _ = cv2.findContours(threshold_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Temukan kontur dengan luas maksimum sebagai pupil
    if contours:
        pupil_contour = max(contours, key=cv2.contourArea)

        # Dapatkan bounding box dari pupil
        x, y, w, h = cv2.boundingRect(pupil_contour)

        # Dapatkan ROI pada pupil
        pupil_roi = eye_roi[y:y + h, x:x + w]

        return pupil_roi

    return None

# Inisialisasi webcam
cap = cv2.VideoCapture(0)

while True:
    # Membaca frame dari webcam
    ret, frame = cap.read()

    # Konversi frame ke skala abu-abu
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah menggunakan Haarcascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Dapatkan ROI wajah
        face_roi = frame[y:y+h, x:x+w]

        # Deteksi mata menggunakan Haarcascades
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

        for (ex, ey, ew, eh) in eyes:
            # Dapatkan ROI mata
            eye_roi = face_roi[ey:ey+eh, ex:ex+ew]

            # Deteksi ROI pada pupil mata
            pupil_roi = detect_pupil_roi(eye_roi)

            # Gambar garis di sekitar pupil
            if pupil_roi is not None:
                cv2.rectangle(eye_roi, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                # Tampilkan hasil deteksi
                cv2.imshow("Pupil ROI with Rectangle", eye_roi)

    # Tampilkan frame utama
    cv2.imshow("Main Frame", frame)

    # Tunggu tombol ESC untuk keluar
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

# Tutup webcam dan jendela OpenCV
cap.release()
cv2.destroyAllWindows()
