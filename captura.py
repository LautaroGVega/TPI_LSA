"""
Captura de muestras — todas las letras + NADA + ESPACIO + FINALIZAR
---------------------------------------------------------------------
Controles:
  Tecla letra  → activa captura de esa letra (A-Z)
  Tecla 0      → captura clase NADA
  Tecla 1      → captura clase ESPACIO
  Tecla 2      → captura clase FINALIZAR
  ESPACIO      → pausar / reanudar
  Q            → guardar y salir
"""

import csv
import math
import os
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH  = "hand_landmarker.task"
DATASET_CSV = "dataset.csv"
CONEXIONES  = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
INTERVALO   = 0.25

LETRAS_VALIDAS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ012")

TECLAS_ESPECIALES = {
    "0": "NADA",
    "1": "ESPACIO",
    "2": "FINALIZAR",
}


def extraer_vector(landmarks) -> list[float]:
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


def cargar_conteos():
    conteos = {}
    if not os.path.exists(DATASET_CSV):
        return conteos
    with open(DATASET_CSV, "r") as f:
        for row in csv.reader(f):
            if row:
                conteos[row[0]] = conteos.get(row[0], 0) + 1
    return conteos


def main():
    print("=" * 56)
    print("  Captura de dataset LSA (vector normalizado)")
    print("=" * 56)
    print("  Tecla letra  → activa captura de esa letra")
    print("  Tecla 0      → NADA")
    print("  Tecla 1      → ESPACIO")
    print("  Tecla 2      → FINALIZAR")
    print("  ESPACIO      → pausar / reanudar")
    print("  Q            → guardar y salir")
    print("=" * 56)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se encontró cámara.")
        return

    conteos      = cargar_conteos()
    letra_activa = None
    capturando   = False
    ultimo_cap   = 0.0
    sesion       = {}

    with crear_detector() as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            resultado = detector.detect(mp_img)

            mano_detectada = bool(resultado.hand_landmarks)
            vector         = None

            if mano_detectada:
                landmarks = resultado.hand_landmarks[0]
                dibujar_mano(frame, landmarks)
                vector = extraer_vector(landmarks)

            ahora = time.time()
            if (capturando and letra_activa and mano_detectada
                    and (ahora - ultimo_cap) >= INTERVALO):
                ultimo_cap = ahora
                with open(DATASET_CSV, "a", newline="") as f:
                    csv.writer(f).writerow([letra_activa] + vector)
                conteos[letra_activa] = conteos.get(letra_activa, 0) + 1
                sesion[letra_activa]  = sesion.get(letra_activa, 0) + 1

            h_frame = frame.shape[0]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 115), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            letra_display = letra_activa if letra_activa else "—"
            color_letra   = (0, 220, 0) if capturando else (0, 160, 220)
            cv2.putText(frame, letra_display, (14, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.8, color_letra, 5, cv2.LINE_AA)

            estado = "CAPTURANDO" if capturando else "EN PAUSA"
            cv2.putText(frame, estado, (110, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_letra, 2, cv2.LINE_AA)

            if letra_activa:
                total_letra  = conteos.get(letra_activa, 0)
                sesion_letra = sesion.get(letra_activa, 0)
                cv2.putText(frame, f"Total: {total_letra}  (+{sesion_letra} hoy)",
                            (110, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1)

            if not mano_detectada:
                cv2.putText(frame, "Sin mano", (110, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 80, 220), 2)

            resumen = "  ".join(f"{k}:{v}" for k, v in sorted(conteos.items()))
            cv2.putText(frame, resumen, (10, h_frame - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
            cv2.putText(frame, "0=NADA 1=ESPACIO 2=FINALIZAR  ESPACIO=pausar  Q=salir",
                        (10, h_frame - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)

            cv2.imshow("Captura LSA", frame)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            elif tecla == ord(" "):
                if letra_activa:
                    capturando = not capturando
            elif tecla != 255:
                char = chr(tecla).upper()
                if char in LETRAS_VALIDAS:
                    nueva_letra = TECLAS_ESPECIALES.get(char, char)
                    if nueva_letra != letra_activa:
                        letra_activa = nueva_letra
                        capturando   = True
                        ultimo_cap   = 0.0
                        print(f"\n  ── Letra activa: {letra_activa} "
                              f"(guardadas: {conteos.get(letra_activa, 0)}) ──")

    cap.release()
    cv2.destroyAllWindows()

    print("\n── Resumen de sesión ──")
    for letra, n in sorted(sesion.items()):
        print(f"  {letra}: +{n} muestras")
    print(f"\n[INFO] Dataset guardado en: {DATASET_CSV}")


if __name__ == "__main__":
    main()