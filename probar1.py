"""
Prueba en tiempo real — Vector 131 valores (2 manos + cara)
--------------------------------------------------------------
Archivos necesarios:
  - hand_landmarker.task
  - blaze_face_short_range.tflite
  - modelo_lsa.keras
  - preproceso.pkl

Uso:
  py -3.12 probar.py
"""

import math
import os
import pickle
import sys
import urllib.request
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

HAND_MODEL  = "hand_landmarker.task"
FACE_MODEL  = "blaze_face_short_range.tflite"
FACE_URL    = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
MODELO_PATH = "modelo_lsa.keras"
PREPROCESO  = "preproceso.pkl"
CONEXIONES  = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS


# ═══════════════════════════════════════════════════════════════════════════
#  VECTOR 131 — idéntico a capturar.py
# ═══════════════════════════════════════════════════════════════════════════
def normalizar_mano(landmarks):
    muneca     = landmarks[0]
    base_medio = landmarks[9]
    distancia = math.sqrt(
        (base_medio.x - muneca.x) ** 2 +
        (base_medio.y - muneca.y) ** 2 +
        (base_medio.z - muneca.z) ** 2
    )
    if distancia == 0:
        distancia = 1.0
    valores = []
    for lm in landmarks:
        valores.extend([
            round((lm.x - muneca.x) / distancia, 4),
            round((lm.y - muneca.y) / distancia, 4),
            round((lm.z - muneca.z) / distancia, 4),
        ])
    return valores, distancia, muneca


def extraer_vector(hand_results, nose_x=None, nose_y=None) -> list[float]:
    manos = hand_results.hand_landmarks
    mano1_vals, escala1, muneca1 = normalizar_mano(manos[0])

    if len(manos) >= 2:
        mano2_vals, _, muneca2 = normalizar_mano(manos[1])
        dist_manos = [
            round((muneca2.x - muneca1.x) / escala1, 4),
            round((muneca2.y - muneca1.y) / escala1, 4),
            round((muneca2.z - muneca1.z) / escala1, 4),
        ]
    else:
        mano2_vals = [0.0] * 63
        dist_manos = [0.0, 0.0, 0.0]

    if nose_x is not None and nose_y is not None:
        pos_cara = [
            round((muneca1.x - nose_x) / escala1, 4),
            round((muneca1.y - nose_y) / escala1, 4),
        ]
    else:
        pos_cara = [0.0, 0.0]

    return mano1_vals + mano2_vals + dist_manos + pos_cara
# ═══════════════════════════════════════════════════════════════════════════


def crear_hand_detector():
    opciones = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
    )
    return mp_vision.HandLandmarker.create_from_options(opciones)


def crear_face_detector():
    opciones = mp_vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL),
        running_mode=mp_vision.RunningMode.IMAGE,
        min_detection_confidence=0.5,
    )
    return mp_vision.FaceDetector.create_from_options(opciones)


def obtener_nariz(face_result):
    if not face_result.detections:
        return None, None
    detection = face_result.detections[0]
    if detection.keypoints and len(detection.keypoints) > 2:
        nose = detection.keypoints[2]
        return nose.x, nose.y
    return None, None


def dibujar_manos(frame, hand_results):
    h, w = frame.shape[:2]
    colores = [(0, 220, 0), (220, 160, 0)]
    for idx, landmarks in enumerate(hand_results.hand_landmarks):
        color = colores[min(idx, 1)]
        puntos = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for conn in CONEXIONES:
            cv2.line(frame, puntos[conn.start], puntos[conn.end], (255, 255, 255), 2)
        for px, py in puntos:
            cv2.circle(frame, (px, py), 5, color, -1)


def main():
    # Descargar modelo de cara si no existe
    if not os.path.exists(FACE_MODEL):
        print(f"[INFO] Descargando modelo de cara...")
        urllib.request.urlretrieve(FACE_URL, FACE_MODEL)

    # Cargar modelo y preprocesamiento
    try:
        modelo = tf.keras.models.load_model(MODELO_PATH)
        print(f"[INFO] Modelo cargado: {MODELO_PATH}")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    try:
        with open(PREPROCESO, "rb") as f:
            datos = pickle.load(f)
        scaler = datos["scaler"]
        clases = datos["clases"]
        print(f"[INFO] Clases: {list(clases)}")
    except FileNotFoundError:
        print(f"[ERROR] No se encontró {PREPROCESO}")
        sys.exit(1)

    # Warmup
    print("[INFO] Calentando modelo...")
    dummy = scaler.transform(np.zeros((1, 131), dtype=np.float32))
    _ = modelo(tf.convert_to_tensor(dummy), training=False)
    print("[INFO] Modelo listo.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se encontró cámara.")
        sys.exit(1)

    print("[INFO] Cámara iniciada. Q=salir\n")

    HISTORIAL_MAX = 10
    historial     = []

    with crear_hand_detector() as hand_det, crear_face_detector() as face_det:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            hand_result = hand_det.detect(mp_img)
            face_result = face_det.detect(mp_img)

            num_manos      = len(hand_result.hand_landmarks)
            mano_detectada = num_manos > 0
            nose_x, nose_y = obtener_nariz(face_result)

            prediccion  = "—"
            confianza   = 0.0
            color_texto = (160, 160, 160)

            if mano_detectada:
                dibujar_manos(frame, hand_result)
                vector = extraer_vector(hand_result, nose_x, nose_y)
                vector_scaled = scaler.transform([vector])

                input_t = tf.convert_to_tensor(vector_scaled, dtype=tf.float32)
                proba   = modelo(input_t, training=False).numpy()[0]
                idx     = np.argmax(proba)
                conf    = proba[idx]

                historial.append(idx)
                if len(historial) > HISTORIAL_MAX:
                    historial.pop(0)
                idx_suave   = max(set(historial), key=historial.count)
                prediccion  = clases[idx_suave]
                confianza   = conf
                color_texto = (0, 220, 0) if confianza >= 0.85 else (0, 180, 220)
            else:
                historial.clear()

            # Dibujar nariz
            if nose_x is not None:
                h, w = frame.shape[:2]
                cv2.circle(frame, (int(nose_x * w), int(nose_y * h)), 8, (220, 0, 220), -1)

            # ── HUD ───────────────────────────────────────────────────────
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

            cv2.putText(frame, prediccion, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.8, color_texto, 5, cv2.LINE_AA)

            if confianza > 0:
                cv2.putText(frame, f"{confianza * 100:.0f}%", (130, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_texto, 3, cv2.LINE_AA)

            barra = int(confianza * 200)
            cv2.rectangle(frame, (10, 88), (210, 98), (60, 60, 60), -1)
            cv2.rectangle(frame, (10, 88), (10 + barra, 98), color_texto, -1)

            cv2.putText(frame, "Q = salir", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

            cv2.imshow("LSA — prediccion en tiempo real", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()