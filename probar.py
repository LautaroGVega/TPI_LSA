"""
Prueba en tiempo real — TensorFlow / Keras
--------------------------------------------
Carga el modelo .keras y predice con la cámara en vivo.

Archivos necesarios en la misma carpeta:
  - hand_landmarker.task
  - modelo_lsa.keras
  - preproceso.pkl

Uso:
  py -3.12 probar.py

Controles:
  Q → salir
"""

import math
import pickle
import sys
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

HAND_MODEL  = "hand_landmarker.task"
MODELO_PATH = "modelo_lsa.keras"
PREPROCESO  = "preproceso.pkl"
CONEXIONES  = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS


# ═══════════════════════════════════════════════════════════════════════════
#  FUNCIÓN CRÍTICA — idéntica a capturar.py
# ═══════════════════════════════════════════════════════════════════════════
def extraer_vector(landmarks) -> list[float]:
    muneca     = landmarks[0]
    base_medio = landmarks[9]

    distancia = math.sqrt(
        (base_medio.x - muneca.x) ** 2 +
        (base_medio.y - muneca.y) ** 2 +
        (base_medio.z - muneca.z) ** 2
    )
    if distancia == 0:
        distancia = 1.0

    vector = []
    for lm in landmarks:
        rel_x = (lm.x - muneca.x) / distancia
        rel_y = (lm.y - muneca.y) / distancia
        rel_z = (lm.z - muneca.z) / distancia
        vector.extend([round(rel_x, 4), round(rel_y, 4), round(rel_z, 4)])
    return vector
# ═══════════════════════════════════════════════════════════════════════════


def crear_detector():
    opciones = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
    )
    return mp_vision.HandLandmarker.create_from_options(opciones)


def dibujar_mano(frame, landmarks):
    h, w = frame.shape[:2]
    puntos = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for conn in CONEXIONES:
        cv2.line(frame, puntos[conn.start], puntos[conn.end], (255, 255, 255), 2)
    for px, py in puntos:
        cv2.circle(frame, (px, py), 5, (0, 220, 0), -1)


def main():
    # ── Cargar modelo y preprocesamiento ──────────────────────────────────
    try:
        modelo = tf.keras.models.load_model(MODELO_PATH)
        print(f"[INFO] Modelo cargado: {MODELO_PATH}")
    except Exception as e:
        print(f"[ERROR] No se pudo cargar {MODELO_PATH}: {e}")
        sys.exit(1)

    try:
        with open(PREPROCESO, "rb") as f:
            datos = pickle.load(f)
        scaler = datos["scaler"]
        clases = datos["clases"]
        print(f"[INFO] Clases del modelo: {list(clases)}")
    except FileNotFoundError:
        print(f"[ERROR] No se encontró {PREPROCESO}")
        sys.exit(1)

    # ── Cámara ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se encontró cámara.")
        sys.exit(1)

    print("[INFO] Cámara iniciada. Presioná Q para salir.")

    HISTORIAL_MAX = 10
    historial     = []

    with crear_detector() as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            resultado = detector.detect(mp_img)

            prediccion  = "—"
            confianza   = 0.0
            color_texto = (160, 160, 160)

            if resultado.hand_landmarks:
                landmarks = resultado.hand_landmarks[0]
                dibujar_mano(frame, landmarks)
                vector = extraer_vector(landmarks)

                # Normalizar con el mismo scaler del entrenamiento
                vector_scaled = scaler.transform([vector])

                # Predecir
                proba = modelo.predict(vector_scaled, verbose=0)[0]
                idx   = np.argmax(proba)
                conf  = proba[idx]

                # Suavizado: historial de últimas N predicciones
                historial.append(idx)
                if len(historial) > HISTORIAL_MAX:
                    historial.pop(0)

                idx_suave   = max(set(historial), key=historial.count)
                prediccion  = clases[idx_suave]
                confianza   = conf
                color_texto = (0, 220, 0) if confianza >= 0.85 else (0, 180, 220)
            else:
                historial.clear()

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