"""
PASO 4 — Validación con video
------------------------------
Procesa un archivo de video frame a frame y muestra la predicción
del modelo sobre cada frame.

Uso:
  python probar-video.py
  python probar-video.py --video ruta/al/video.mp4

Por defecto usa video.mp4 en la misma carpeta que este script.

Opciones:
  --video   Ruta al archivo de video (mp4, avi, mov, etc.)
  --salida  Ruta para guardar el video procesado (opcional)
             Ejemplo: --salida resultado.mp4

Controles durante la reproducción:
  ESPACIO  → pausar / reanudar
  Q        → salir
"""

import argparse
import math
import pickle
import sys
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH   = "hand_landmarker.task"
MODELO_PATH  = "modelo_lsa.pkl"
VIDEO_DEFAULT = Path(__file__).resolve().parent / "video.mp4"
CONEXIONES  = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS


# ═══════════════════════════════════════════════════════════════════════════
#  FUNCIÓN CRÍTICA — idéntica a 1_capturar.py y 3_probar.py
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


def procesar_video(ruta_entrada: str, ruta_salida: str | None):
    # Cargar modelo
    try:
        with open(MODELO_PATH, "rb") as f:
            datos = pickle.load(f)
        modelo  = datos["modelo"]
        encoder = datos["encoder"]
    except FileNotFoundError:
        print(f"[ERROR] No se encontró {MODELO_PATH}. Ejecutá primero 2_entrenar.py")
        sys.exit(1)

    print(f"[INFO] Clases del modelo: {list(encoder.classes_)}")

    # Abrir video
    cap = cv2.VideoCapture(ruta_entrada)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir el video: {ruta_entrada}")
        sys.exit(1)

    fps        = cap.get(cv2.CAP_PROP_FPS) or 30
    ancho      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video: {ancho}x{alto} @ {fps:.1f} fps — {total_frames} frames")

    # Writer para guardar resultado (opcional)
    writer = None
    if ruta_salida:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(ruta_salida, fourcc, fps, (ancho, alto))
        print(f"[INFO] Guardando resultado en: {ruta_salida}")

    # Historial para suavizar predicciones
    HISTORIAL_MAX = 8
    historial     = []
    pausado       = False
    frame_actual  = 0

    # Estadísticas
    stats = {}

    with crear_detector() as detector:
        while True:
            if not pausado:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_actual += 1

            prediccion  = "—"
            confianza   = 0.0
            color_texto = (160, 160, 160)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            resultado = detector.detect(mp_img)

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

                idx_suave  = max(set(historial), key=historial.count)
                prediccion = encoder.classes_[idx_suave]
                confianza  = conf
                color_texto = (0, 220, 0) if confianza >= 0.85 else (0, 180, 220)

                # Acumular estadísticas
                stats[prediccion] = stats.get(prediccion, 0) + 1
            else:
                historial.clear()

            # ── HUD ───────────────────────────────────────────────────────────
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (ancho, 105), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

            # Letra predicha grande
            cv2.putText(frame, prediccion, (15, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.8, color_texto, 5, cv2.LINE_AA)

            # Confianza
            if confianza > 0:
                cv2.putText(frame, f"{confianza * 100:.0f}%", (135, 78),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, color_texto, 3, cv2.LINE_AA)

            # Barra de confianza
            barra = int(confianza * 180)
            cv2.rectangle(frame, (10, 90), (190, 100), (60, 60, 60), -1)
            cv2.rectangle(frame, (10, 90), (10 + barra, 100), color_texto, -1)

            # Progreso del video
            progreso_ratio = frame_actual / total_frames if total_frames else 0
            barra_prog     = int(progreso_ratio * (ancho - 20))
            cv2.rectangle(frame, (10, alto - 18), (ancho - 10, alto - 8), (60, 60, 60), -1)
            cv2.rectangle(frame, (10, alto - 18), (10 + barra_prog, alto - 8), (180, 180, 180), -1)

            # Frame counter y estado pausa
            estado_txt = "|| PAUSADO" if pausado else f"Frame {frame_actual}/{total_frames}"
            cv2.putText(frame, estado_txt, (10, alto - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            cv2.putText(frame, "ESPACIO=pausar  Q=salir",
                        (ancho - 220, alto - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

            cv2.imshow("LSA — validacion con video", frame)

            if writer:
                writer.write(frame)

            # Controles — esperar según fps para reproducción correcta
            espera = max(1, int(1000 / fps))
            tecla  = cv2.waitKey(espera) & 0xFF

            if tecla == ord("q"):
                break
            elif tecla == ord(" "):
                pausado = not pausado
                print(f"  {'PAUSADO' if pausado else 'REANUDADO'} en frame {frame_actual}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # Resumen final
    print("\n--- Resumen de predicciones ---")
    total_con_mano = sum(stats.values())
    for letra, n in sorted(stats.items(), key=lambda x: -x[1]):
        pct = n / total_con_mano * 100 if total_con_mano else 0
        barra = "#" * int(pct / 2)
        print(f"  {letra:>6}: {n:>4} frames ({pct:4.1f}%)  {barra}")
    print(f"  {'TOTAL':>6}: {total_con_mano:>4} frames con mano detectada")

    if ruta_salida:
        print(f"\n[INFO] Video guardado en: {ruta_salida}")


def main():
    parser = argparse.ArgumentParser(description="Validación del modelo LSA con video")
    parser.add_argument("--video", "-v", default=str(VIDEO_DEFAULT),
                        help=f"Ruta al video de entrada (por defecto: {VIDEO_DEFAULT.name})")
    parser.add_argument("--salida", "-s", default=None,
                        help="Ruta para guardar el video con anotaciones (opcional)")
    args = parser.parse_args()

    procesar_video(args.video, args.salida)


if __name__ == "__main__":
    main()