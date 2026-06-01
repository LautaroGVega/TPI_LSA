"""
PASO 3 — Prueba en tiempo real
--------------------------------
Carga el modelo entrenado y predice en tiempo real con la cámara.

IMPORTANTE: usa exactamente la MISMA función extraer_vector() que
1_capturar.py. Si las dos no son idénticas, el modelo no funciona.
"""

import math
import pickle
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH  = "hand_landmarker.task"
MODELO_PATH = "modelo_lsa.pkl"
CONEXIONES  = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS


# ═══════════════════════════════════════════════════════════════════════════
#  FUNCIÓN CRÍTICA — debe ser idéntica en 1_capturar.py y 3_probar.py
# ═══════════════════════════════════════════════════════════════════════════
def extraer_vector(landmarks) -> list[float]:
    """
    Devuelve un vector de 63 valores normalizados:
      1. Invariancia de traslación: se resta la muñeca (landmark 0).
      2. Invariancia de escala: se divide por la distancia muñeca → base del
         dedo medio (landmark 9).
    """
    muneca = landmarks[0]
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
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
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
    try:
        with open(MODELO_PATH, "rb") as f:
            datos = pickle.load(f)
        modelo  = datos["modelo"]
        encoder = datos["encoder"]
    except FileNotFoundError:
        print(f"[ERROR] No se encontró {MODELO_PATH}. Ejecutá primero 2_entrenar.py")
        return

    print(f"[INFO] Clases del modelo: {list(encoder.classes_)}")
    print("[INFO] Cámara iniciada. Presioná Q para salir.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se encontró cámara.")
        return

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

                proba = modelo.predict_proba([vector])[0]
                idx   = np.argmax(proba)
                conf  = proba[idx]

                historial.append(idx)
                if len(historial) > HISTORIAL_MAX:
                    historial.pop(0)

                idx_suave   = max(set(historial), key=historial.count)
                prediccion  = encoder.classes_[idx_suave]
                confianza   = conf
                color_texto = (0, 220, 0) if confianza >= 0.85 else (0, 180, 220)
            else:
                historial.clear()

            # HUD
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

            cv2.imshow("LSA — predicción en tiempo real", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()