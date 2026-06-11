"""
Cliente LSA — Recibe stream del Rapiro, procesa y llama a Lambda
------------------------------------------------------------------
Corre en tu notebook. Recibe video del Pi, corre MediaPipe,
manda el vector a Lambda para predicción, y muestra el resultado.

Instalar:
  pip install boto3 mediapipe opencv-python numpy

Configurar AWS:
  aws configure  (poner Access Key, Secret, region us-east-2)

Uso:
  py -3.12 cliente_lsa.py --pi 192.168.1.XX

  Si querés probar sin el Pi (usando tu cámara local):
  py -3.12 cliente_lsa.py --local
"""

import argparse
import json
import math
import os
import sys
import time
import urllib.request
import cv2
import numpy as np
import boto3
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

HAND_MODEL  = "hand_landmarker.task"
FACE_MODEL  = "blaze_face_short_range.tflite"
FACE_URL    = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
CONEXIONES  = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

LAMBDA_NAME = "rapiro-lsa-inference"
AWS_REGION  = "us-east-2"
WEB_URL     = "https://rapiro.onrender.com"

TIEMPO_CONFIRMACION = 1.0
CONFIANZA_MINIMA    = 0.80
HISTORIAL_MAX       = 10


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


def extraer_vector(hand_results, nose_x=None, nose_y=None):
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
    return mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.7,
        ))


def crear_face_detector():
    return mp_vision.FaceDetector.create_from_options(
        mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL),
            running_mode=mp_vision.RunningMode.IMAGE,
            min_detection_confidence=0.5,
        ))


def obtener_nariz(face_result):
    if not face_result.detections:
        return None, None
    d = face_result.detections[0]
    if d.keypoints and len(d.keypoints) > 2:
        return d.keypoints[2].x, d.keypoints[2].y
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


def llamar_lambda(lambda_client, vector, session_id, texto_acumulado="", finalizar=False):
    """Invoca Lambda y devuelve (letra, confianza)."""
    payload = {
        "vector": vector,
        "SessionId": session_id,
        "Source": "cliente_lsa",
        "texto_acumulado": texto_acumulado,
        "finalizar": finalizar,
    }
    try:
        response = lambda_client.invoke(
            FunctionName=LAMBDA_NAME,
            InvocationType="RequestResponse",
            Payload=json.dumps(payload),
        )
        result = json.loads(response["Payload"].read())
        if isinstance(result.get("body"), str):
            body = json.loads(result["body"])
        else:
            body = result
        return body.get("letra", "?"), body.get("confianza", 0.0)
    except Exception as e:
        print(f"  [LAMBDA ERROR] {e}")
        return "?", 0.0


def enviar_web(texto):
    try:
        body = json.dumps({"texto": texto}).encode("utf-8")
        req = urllib.request.Request(
            f"{WEB_URL}/api/finalizar", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
        print(f"  [WEB] Enviado a {WEB_URL}")
    except Exception as e:
        print(f"  [WEB] Error: {e}")


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--pi", type=str, help="IP del Rapiro (ej: 192.168.1.50)")
    parser.add_argument("--local", action="store_true", help="Usar cámara local en vez del Pi")
    args = parser.parse_args()

    if not args.pi and not args.local:
        print("Uso:")
        print("  py -3.12 cliente_lsa.py --pi 192.168.1.XX   (stream del Rapiro)")
        print("  py -3.12 cliente_lsa.py --local              (cámara local)")
        sys.exit(1)

    # Descargar modelo de cara si falta
    if not os.path.exists(FACE_MODEL):
        print("[INFO] Descargando modelo de cara...")
        urllib.request.urlretrieve(FACE_URL, FACE_MODEL)

    # Conectar a AWS Lambda
    print("[INFO] Conectando a AWS Lambda...")
    lambda_client = boto3.client("lambda", region_name=AWS_REGION)
    session_id = f"session-{int(time.time())}"
    print(f"[INFO] Session: {session_id}")

    # Abrir video
    if args.local:
        print("[INFO] Usando cámara local")
        cap = cv2.VideoCapture(0)
    else:
        stream_url = f"http://{args.pi}:8080/video"
        print(f"[INFO] Conectando a Rapiro: {stream_url}")
        cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la fuente de video")
        sys.exit(1)

    print("[INFO] Video conectado. Q=salir\n")

    texto_formado     = ""
    historial         = []
    letra_candidata   = None
    tiempo_inicio     = 0.0
    letra_confirmada  = False
    cooldown_hasta    = 0.0
    ultimo_lambda     = 0.0
    ultima_prediccion = "-"
    ultima_confianza  = 0.0

    with crear_hand_detector() as hand_det, crear_face_detector() as face_det:
        while True:
            ok, frame = cap.read()
            if not ok:
                if not args.local:
                    # Reintentar conexión al stream
                    cap = cv2.VideoCapture(f"http://{args.pi}:8080/video")
                    continue
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            hand_result = hand_det.detect(mp_img)
            face_result = face_det.detect(mp_img)

            num_manos      = len(hand_result.hand_landmarks)
            mano_detectada = num_manos > 0
            nose_x, nose_y = obtener_nariz(face_result)

            ahora      = time.time()
            prediccion = ultima_prediccion
            confianza  = ultima_confianza
            color_pred = (160, 160, 160)
            progreso   = 0.0

            if mano_detectada:
                dibujar_manos(frame, hand_result)
                vector = extraer_vector(hand_result, nose_x, nose_y)

                # Llamar a Lambda cada 0.4 segundos
                if ahora - ultimo_lambda >= 0.4:
                    prediccion, confianza = llamar_lambda(
                        lambda_client, vector, session_id
                    )
                    ultima_prediccion = prediccion
                    ultima_confianza  = confianza
                    ultimo_lambda     = ahora

                    historial.append(prediccion)
                    if len(historial) > HISTORIAL_MAX:
                        historial.pop(0)
                    prediccion = max(set(historial), key=historial.count)

                if prediccion == "NADA":
                    letra_candidata  = None
                    tiempo_inicio    = 0.0
                    letra_confirmada = False

                elif confianza >= CONFIANZA_MINIMA and ahora > cooldown_hasta:
                    color_pred = (0, 180, 220)

                    if prediccion == letra_candidata:
                        transcurrido = ahora - tiempo_inicio
                        progreso = min(transcurrido / TIEMPO_CONFIRMACION, 1.0)

                        if progreso >= 1.0 and not letra_confirmada:
                            if prediccion == "FINALIZAR":
                                print(f"\n  ★ FINALIZADO: \"{texto_formado}\"")
                                # Llamar Lambda con flag finalizar para Polly
                                llamar_lambda(lambda_client, vector, session_id,
                                            texto_formado, finalizar=True)
                                enviar_web(texto_formado)

                                for _ in range(60):
                                    ok2, f2 = cap.read()
                                    if ok2:
                                        ov = f2.copy()
                                        cv2.rectangle(ov, (0,0), (f2.shape[1], f2.shape[0]), (0,0,0), -1)
                                        cv2.addWeighted(ov, 0.7, f2, 0.3, 0, f2)
                                        cv2.putText(f2, "ENVIADO", (60, 180),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 220, 0), 5)
                                        cv2.putText(f2, texto_formado, (40, 280),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)
                                        cv2.imshow("RAPIRO LSA — Cloud", f2)
                                    cv2.waitKey(33)

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

            # Nariz
            if nose_x is not None:
                h, w = frame.shape[:2]
                cv2.circle(frame, (int(nose_x * w), int(nose_y * h)), 8, (220, 0, 220), -1)

            # HUD
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 110), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            cv2.putText(frame, str(prediccion), (15, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, color_pred, 5, cv2.LINE_AA)

            if confianza > 0:
                cv2.putText(frame, f"{confianza*100:.0f}%", (140, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_pred, 2, cv2.LINE_AA)

            # Cloud indicator
            cv2.putText(frame, "CLOUD", (frame.shape[1] - 90, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 0, 220), 2, cv2.LINE_AA)

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
            cv2.imshow("RAPIRO LSA — Cloud", frame)

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