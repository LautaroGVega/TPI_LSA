"""
Captura de muestras LSA — Vector de 131 valores
--------------------------------------------------
Soporta: dos manos + posición facial

Vector: 63 (mano 1) + 63 (mano 2 o ceros) + 3 (distancia manos) + 2 (cara)

Controles:
  Tecla letra  → captura esa letra (A-Z)
  Tecla 0      → NADA
  Tecla 1      → ESPACIO
  Tecla 2      → FINALIZAR 
  ESPACIO      → pausar / reanudar
  Q            → guardar y salir

Modelos necesarios (se descargan solos):
  - hand_landmarker.task
  - blaze_face_short_range.tflite
"""

import csv
import math
import os
import time
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

HAND_MODEL_PATH = "hand_landmarker.task"
FACE_MODEL_PATH = "blaze_face_short_range.tflite"
FACE_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"

DATASET_CSV = "dataset.csv"
CONEXIONES  = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
INTERVALO   = 0.25

LETRAS_VALIDAS = set("ABCDEFGHIJKLMNÑOPQRSTUVWXYZ012")
TECLAS_ESPECIALES = {
    "0": "NADA",
    "1": "ESPACIO",
    "2": "FINALIZAR",
}


def descargar_modelo_cara():
    if os.path.exists(FACE_MODEL_PATH):
        return
    print(f"[INFO] Descargando modelo de cara...")
    urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)
    print(f"[INFO] Guardado: {FACE_MODEL_PATH}")


# ═══════════════════════════════════════════════════════════════════════════
#  VECTOR DE 131 VALORES
#  [0-62]   Mano dominante: 21 puntos × 3 coords, normalizado
#  [63-125] Mano secundaria: igual, o 63 ceros si no hay
#  [126-128] Distancia entre muñecas (x, y, z), o ceros
#  [129-130] Posición muñeca→nariz (x, y), o ceros
# ═══════════════════════════════════════════════════════════════════════════
def normalizar_mano(landmarks):
    """Devuelve 63 valores normalizados de una mano."""
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
    """
    Construye el vector de 131 valores a partir de los resultados de
    detección de manos y la posición de la nariz.
    """
    manos = hand_results.hand_landmarks
    num_manos = len(manos)

    # ── Mano dominante (siempre presente si se llama a esta función) ──
    mano1_vals, escala1, muneca1 = normalizar_mano(manos[0])

    # ── Mano secundaria ──
    if num_manos >= 2:
        mano2_vals, escala2, muneca2 = normalizar_mano(manos[1])

        # Distancia entre muñecas (normalizada por escala de mano dominante)
        dist_manos = [
            round((muneca2.x - muneca1.x) / escala1, 4),
            round((muneca2.y - muneca1.y) / escala1, 4),
            round((muneca2.z - muneca1.z) / escala1, 4),
        ]
    else:
        mano2_vals = [0.0] * 63
        dist_manos = [0.0, 0.0, 0.0]

    # ── Posición facial ──
    if nose_x is not None and nose_y is not None:
        pos_cara = [
            round((muneca1.x - nose_x) / escala1, 4),
            round((muneca1.y - nose_y) / escala1, 4),
        ]
    else:
        pos_cara = [0.0, 0.0]

    # Total: 63 + 63 + 3 + 2 = 131
    return mano1_vals + mano2_vals + dist_manos + pos_cara
# ═══════════════════════════════════════════════════════════════════════════


def crear_hand_detector():
    opciones = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,                          # ← detecta hasta 2 manos
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
    )
    return mp_vision.HandLandmarker.create_from_options(opciones)


def crear_face_detector():
    opciones = mp_vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        min_detection_confidence=0.5,
    )
    return mp_vision.FaceDetector.create_from_options(opciones)


def dibujar_manos(frame, hand_results):
    h, w = frame.shape[:2]
    colores_mano = [(0, 220, 0), (220, 160, 0)]  # verde = mano 1, amarillo = mano 2

    for idx, landmarks in enumerate(hand_results.hand_landmarks):
        color = colores_mano[min(idx, 1)]
        puntos = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for conn in CONEXIONES:
            cv2.line(frame, puntos[conn.start], puntos[conn.end], (255, 255, 255), 2)
        for px, py in puntos:
            cv2.circle(frame, (px, py), 5, color, -1)


def obtener_nariz(face_result):
    if not face_result.detections:
        return None, None
    detection = face_result.detections[0]
    if detection.keypoints and len(detection.keypoints) > 2:
        nose = detection.keypoints[2]
        return nose.x, nose.y
    return None, None


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
    descargar_modelo_cara()

    print("=" * 58)
    print("  Captura LSA — 131 valores (2 manos + cara)")
    print("=" * 58)
    print("  Tecla letra  → captura esa letra")
    print("  0=NADA  1=ESPACIO  2=FINALIZAR")
    print("  ESPACIO=pausar  Q=salir")
    print("=" * 58)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se encontró cámara.")
        return

    conteos      = cargar_conteos()
    letra_activa = None
    capturando   = False
    ultimo_cap   = 0.0
    sesion       = {}

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
            vector         = None

            if mano_detectada:
                dibujar_manos(frame, hand_result)
                vector = extraer_vector(hand_result, nose_x, nose_y)

            # Dibujar nariz
            if nose_x is not None:
                h, w = frame.shape[:2]
                nx, ny = int(nose_x * w), int(nose_y * h)
                cv2.circle(frame, (nx, ny), 8, (220, 0, 220), -1, cv2.LINE_AA)

            # Captura automática
            ahora = time.time()
            if (capturando and letra_activa and mano_detectada
                    and (ahora - ultimo_cap) >= INTERVALO):
                ultimo_cap = ahora
                with open(DATASET_CSV, "a", newline="") as f:
                    csv.writer(f).writerow([letra_activa] + vector)
                conteos[letra_activa] = conteos.get(letra_activa, 0) + 1
                sesion[letra_activa]  = sesion.get(letra_activa, 0) + 1

            # ── HUD ───────────────────────────────────────────────────────
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

            # Indicadores
            manos_txt = f"Manos: {num_manos}" if mano_detectada else "Sin mano"
            manos_col = (0, 220, 0) if num_manos == 2 else (0, 180, 220) if num_manos == 1 else (0, 80, 220)
            cv2.putText(frame, manos_txt, (110, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, manos_col, 1, cv2.LINE_AA)

            cara_txt = "Cara: SI" if nose_x is not None else "Cara: NO"
            cara_col = (0, 220, 0) if nose_x is not None else (0, 80, 220)
            cv2.putText(frame, cara_txt, (200, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, cara_col, 1, cv2.LINE_AA)

            cv2.putText(frame, f"Vector: 131 val", (300, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1, cv2.LINE_AA)

            resumen = "  ".join(f"{k}:{v}" for k, v in sorted(conteos.items()))
            cv2.putText(frame, resumen, (10, h_frame - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
            cv2.putText(frame, "0=NADA 1=ESPACIO 2=FIN  ESPACIO=pausar  Q=salir",
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
                        print(f"\n  ── Letra: {letra_activa} "
                              f"(guardadas: {conteos.get(letra_activa, 0)}) ──")

    cap.release()
    cv2.destroyAllWindows()

    print("\n── Resumen ──")
    for letra, n in sorted(sesion.items()):
        print(f"  {letra}: +{n}")
    print(f"\n[INFO] Dataset: {DATASET_CSV}")
    print(f"[INFO] Vector: 131 valores (63+63+3+2)")


if __name__ == "__main__":
    main()