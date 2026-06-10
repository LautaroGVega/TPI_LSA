"""
RAPIRO LSA — Detecta señas y al FINALIZAR manda a la web
-----------------------------------------------------------
La cámara corre normal formando la oración letra por letra.
Cuando detecta FINALIZAR, envía el texto completo a la web.

Uso:
  Terminal 1:  py -3.12 servidor_web.py
  Terminal 2:  py -3.12 rapiro_lsa.py
  Navegador:   http://localhost:5000
"""

import math
import pickle
import sys
import time
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request
import json

HAND_MODEL  = "hand_landmarker.task"
MODELO_PATH = "modelo_lsa.keras"
PREPROCESO  = "preproceso.pkl"
CONEXIONES  = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

TIEMPO_CONFIRMACION = 1.0
CONFIANZA_MINIMA    = 0.80
HISTORIAL_MAX       = 10
WEB_URL             = "https://rapiro.onrender.com"


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
        vector.extend([
            round((lm.x - muneca.x) / distancia, 4),
            round((lm.y - muneca.y) / distancia, 4),
            round((lm.z - muneca.z) / distancia, 4),
        ])
    return vector


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


def enviar_a_web(texto):
    """Envía el mensaje finalizado a la página web."""
    try:
        body = json.dumps({"texto": texto}).encode("utf-8")
        req = urllib.request.Request(
            f"{WEB_URL}/api/finalizar",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=3)
        print(f"  [WEB] Mensaje enviado a {WEB_URL}")
    except Exception as e:
        print(f"  [WEB] No se pudo enviar: {e}")


def dibujar_barra(frame, x, y, ancho, alto, progreso, color):
    cv2.rectangle(frame, (x, y), (x + ancho, y + alto), (60, 60, 60), -1)
    relleno = int(progreso * ancho)
    if relleno > 0:
        cv2.rectangle(frame, (x, y), (x + relleno, y + alto), color, -1)


def dibujar_texto(frame, texto):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 70), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, "Texto:", (10, h - 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
    mostrar = texto if texto else "|"
    max_c = (w - 20) // 18
    if len(mostrar) > max_c:
        mostrar = "..." + mostrar[-(max_c - 3):]
    cv2.putText(frame, mostrar, (10, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    # Cargar modelo
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
    dummy = scaler.transform(np.zeros((1, 63), dtype=np.float32))
    _ = modelo(tf.convert_to_tensor(dummy), training=False)
    print("[INFO] Modelo listo.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se encontró cámara.")
        sys.exit(1)

    print("[INFO] Cámara iniciada.")
    print("[INFO] BACKSPACE=borrar  ESC=limpiar  Q=salir\n")

    texto_formado     = ""
    historial         = []
    letra_candidata   = None
    tiempo_inicio     = 0.0
    letra_confirmada  = False
    cooldown_hasta    = 0.0

    with crear_detector() as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            resultado = detector.detect(mp_img)

            ahora      = time.time()
            prediccion = None
            confianza  = 0.0
            color_pred = (160, 160, 160)
            progreso   = 0.0

            if resultado.hand_landmarks:
                landmarks = resultado.hand_landmarks[0]
                dibujar_mano(frame, landmarks)
                vector = extraer_vector(landmarks)
                vector_scaled = scaler.transform([vector])

                # Predicción rápida
                input_t = tf.convert_to_tensor(vector_scaled, dtype=tf.float32)
                proba   = modelo(input_t, training=False).numpy()[0]
                idx     = np.argmax(proba)
                conf    = proba[idx]

                historial.append(idx)
                if len(historial) > HISTORIAL_MAX:
                    historial.pop(0)
                idx_suave  = max(set(historial), key=historial.count)
                prediccion = clases[idx_suave]
                confianza  = conf

                if prediccion == "NADA":
                    letra_candidata  = None
                    tiempo_inicio    = 0.0
                    letra_confirmada = False
                    color_pred = (160, 160, 160)

                elif confianza >= CONFIANZA_MINIMA and ahora > cooldown_hasta:
                    color_pred = (0, 180, 220)

                    if prediccion == letra_candidata:
                        transcurrido = ahora - tiempo_inicio
                        progreso = min(transcurrido / TIEMPO_CONFIRMACION, 1.0)

                        if progreso >= 1.0 and not letra_confirmada:

                            if prediccion == "FINALIZAR":
                                # ══════ FINALIZAR: manda a la web ══════
                                print(f"\n  ★ FINALIZADO: \"{texto_formado}\"")
                                enviar_a_web(texto_formado)

                                # Pantalla de confirmación 3 segundos
                                for _ in range(90):
                                    ok2, f2 = cap.read()
                                    if ok2:
                                        ov = f2.copy()
                                        cv2.rectangle(ov, (0,0), (f2.shape[1], f2.shape[0]), (0,0,0), -1)
                                        cv2.addWeighted(ov, 0.7, f2, 0.3, 0, f2)
                                        cv2.putText(f2, "ENVIADO A LA WEB", (40, 180),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 220, 0), 4, cv2.LINE_AA)
                                        cv2.putText(f2, texto_formado, (40, 280),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4, cv2.LINE_AA)
                                        cv2.imshow("RAPIRO LSA", f2)
                                    cv2.waitKey(33)

                                # Reset para nueva oración
                                texto_formado    = ""
                                letra_candidata  = None
                                letra_confirmada = False
                                cooldown_hasta   = ahora + 2.0
                                historial.clear()
                                continue

                            elif prediccion == "ESPACIO":
                                texto_formado += " "
                            else:
                                texto_formado += prediccion

                            letra_confirmada = True
                            cooldown_hasta   = ahora + 0.8
                            color_pred = (0, 220, 0)
                            print(f"  ✓ {prediccion}  →  \"{texto_formado}\"")

                        elif letra_confirmada:
                            color_pred = (0, 220, 0)
                            progreso = 1.0
                    else:
                        letra_candidata  = prediccion
                        tiempo_inicio    = ahora
                        letra_confirmada = False
                else:
                    letra_candidata  = None
                    tiempo_inicio    = 0.0
                    letra_confirmada = False
            else:
                historial.clear()
                letra_candidata  = None
                tiempo_inicio    = 0.0
                letra_confirmada = False

            # ── HUD ───────────────────────────────────────────────────────
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 110), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            letra_mostrar = prediccion if prediccion else "—"
            cv2.putText(frame, letra_mostrar, (15, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, color_pred, 5, cv2.LINE_AA)

            if confianza > 0 and prediccion:
                cv2.putText(frame, f"{confianza*100:.0f}%", (140, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_pred, 2, cv2.LINE_AA)

            if letra_confirmada:
                est, col = "CONFIRMADO", (0, 220, 0)
            elif letra_candidata and prediccion != "NADA":
                est, col = "Mantene...", (0, 180, 220)
            else:
                est, col = "Esperando", (160, 160, 160)

            cv2.putText(frame, est, (140, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
            dibujar_barra(frame, 140, 80, 200, 14, progreso, color_pred)

            if 0 < progreso < 1.0:
                cv2.putText(frame, f"{TIEMPO_CONFIRMACION*(1-progreso):.1f}s",
                            (350, 93), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

            cv2.putText(frame, "BACKSPACE=borrar  ESC=limpiar  Q=salir",
                        (10, frame.shape[0] - 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

            dibujar_texto(frame, texto_formado)
            cv2.imshow("RAPIRO LSA", frame)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            elif tecla == 8 and texto_formado:
                texto_formado = texto_formado[:-1]
            elif tecla == 27:
                texto_formado = ""

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()